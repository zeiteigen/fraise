"""The Fraise pretraining loop for the project's concrete model and data."""

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import platform
import random

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from components import device
from components.model import FraiseModel, ModelConfig
from datasets_mgr import (
    get_dataset_fingerprint,
    get_dataset_name,
    get_dataset_size_bytes,
    get_dataset_sources,
)
from helpers.get_git_commit import get_git_snapshot
from tokenizer import get_tokenizer_fingerprint

from .hyperparameters import (
    TrainingHyperparameters,
    TrainingSettings,
    allocate_dataset_token_budgets,
    _validate_hyperparameters,
)
from .packing import (
    PackedTokenBatch,
    iter_dataset_packed_batches,
    iter_packed_batches,
    pretokenize_split,
    special_token_id,
)
from .save_mgr import SaveManager


@dataclass
class TrainingState:
    """Counters needed to resume within an epoch.

    SaveManager stores and restores RNG state alongside these counters; RNG
    state is deliberately not part of this small bookkeeping object.
    """

    epoch: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    tokens_seen: int = 0
    epoch_tokens_seen: int = 0
    epoch_loss_sum: float = 0.0
    epoch_valid_tokens: int = 0
    optimizer_steps_in_epoch: int = 0

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, object]) -> "TrainingState":
        saved = checkpoint.get("training_state")
        if not isinstance(saved, Mapping):
            raise ValueError("checkpoint is missing training_state")

        return cls(
            epoch=int(saved["epoch"]),
            next_batch_index=int(saved["next_batch_index"]),
            global_step=int(saved["global_step"]),
            tokens_seen=int(saved["tokens_seen"]),
            epoch_tokens_seen=int(saved["epoch_tokens_seen"]),
            epoch_loss_sum=float(saved["epoch_loss_sum"]),
            epoch_valid_tokens=int(saved["epoch_valid_tokens"]),
            optimizer_steps_in_epoch=int(saved["optimizer_steps_in_epoch"]),
        )

    def start_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.next_batch_index = 0
        self.epoch_tokens_seen = 0
        self.epoch_loss_sum = 0.0
        self.epoch_valid_tokens = 0
        self.optimizer_steps_in_epoch = 0

    def record_step(
        self,
        batch_index: int,
        loss: float,
        valid_tokens: int,
    ) -> None:
        self.global_step += 1
        self.tokens_seen += valid_tokens
        self.epoch_tokens_seen += valid_tokens
        self.next_batch_index = batch_index + 1
        self.epoch_loss_sum += loss * valid_tokens
        self.epoch_valid_tokens += valid_tokens
        self.optimizer_steps_in_epoch += 1

    def finish_epoch(self) -> None:
        self.epoch += 1
        self.next_batch_index = 0
        self.epoch_tokens_seen = 0
        self.epoch_loss_sum = 0.0
        self.epoch_valid_tokens = 0
        self.optimizer_steps_in_epoch = 0

    def as_checkpoint(self) -> dict[str, object]:
        return asdict(self)


def _loss_for_batch(
    model: FraiseModel,
    batch: PackedTokenBatch,
) -> tuple[torch.Tensor, int] | None:
    """Return the mean cross-entropy over non-padding targets."""
    logits = model(batch.input_ids)

    valid_targets = batch.target_mask.reshape(-1)
    valid_count = int(valid_targets.sum().item())
    if valid_count == 0:
        return None

    flat_targets = batch.targets.reshape(-1)
    selected_targets = flat_targets[valid_targets]
    if int(selected_targets.min().item()) < 0 or int(
        selected_targets.max().item()
    ) >= logits.size(-1):
        raise ValueError("target token ID is outside the model vocabulary")

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1))[valid_targets],
        selected_targets,
    )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite loss detected")
    return loss, valid_count


def _autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _move_batch(
    batch: PackedTokenBatch,
    target_device: torch.device,
) -> PackedTokenBatch:
    return PackedTokenBatch(
        input_ids=batch.input_ids.to(target_device, non_blocking=True),
        targets=batch.targets.to(target_device, non_blocking=True),
        target_mask=batch.target_mask.to(target_device, non_blocking=True),
    )


def _evaluate(
    model: FraiseModel,
    split: Dataset,
    settings: TrainingSettings,
    pad_token_id: int,
) -> float | None:
    """Compute token-weighted loss on the validation split."""
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for _, cpu_batch in iter_packed_batches(
            split,
            settings.sequence_length,
            settings.packing_stride,
            pad_token_id,
            batch_size=settings.batch_size,
            seed=settings.seed,
            shuffle=False,
            max_tokens=settings.max_validation_tokens,
        ):
            batch = _move_batch(cpu_batch, model_device)
            with _autocast_context(settings.use_bf16):
                result = _loss_for_batch(model, batch)
            if result is None:
                continue

            loss, valid_tokens = result
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens

    return total_loss / total_tokens if total_tokens else None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def _dataset_manifest(dataset: DatasetDict) -> dict[str, object]:
    splits: dict[str, object] = {}
    for split_name, split_dataset in sorted(dataset.items()):
        splits[split_name] = {
            "rows": len(split_dataset),
            "size_in_bytes": get_dataset_size_bytes(split_dataset),
            "fingerprint": get_dataset_fingerprint(split_dataset),
        }
    return {
        "name": get_dataset_name(dataset),
        "fingerprint": get_dataset_fingerprint(dataset),
        "splits": splits,
    }


def _run_manifest(
    dataset: DatasetDict,
    tokenizer: Tokenizer,
    model_config: Mapping[str, object],
    settings: TrainingSettings,
) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "git": get_git_snapshot(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "precision": "bf16-mixed" if settings.use_bf16 else "fp32",
        "model_config": dict(model_config),
        "training": asdict(settings),
        "dataset": _dataset_manifest(dataset),
        "tokenizer": {
            "fingerprint": get_tokenizer_fingerprint(tokenizer),
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token_id": special_token_id(tokenizer, "<|pad|>"),
            "bos_token_id": special_token_id(tokenizer, "<|bos|>"),
            "eos_token_id": special_token_id(tokenizer, "<|eos|>"),
        },
    }


def pretrain_transformer(
    tokenizer: Tokenizer,
    dataset: DatasetDict,
    hyperparameters: TrainingHyperparameters,
    model_config: ModelConfig,
    save_path: str,
    model_name: str = "fraise_model",
    resume_from: str | None = None,
    config_paths: Iterable[str | Path] | str | Path | None = None,
) -> FraiseModel:
    """Train FraiseModel on the project's packed text dataset."""
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "pretrain_transformer expects the DatasetDict returned by "
            "config.get_dataset_loader()"
        )
    if "train" not in dataset:
        raise ValueError("dataset must contain a 'train' split")

    raw_train_sources = get_dataset_sources(dataset, "train")
    settings = _validate_hyperparameters(
        hyperparameters,
        model_config,
        raw_train_sources.keys(),
    )
    if settings.use_bf16 and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "use_bf16=True requires a CUDA device with BF16 support; "
            f"got {device}"
        )
    if settings.use_bf16:
        torch.set_float32_matmul_precision("high")

    training_steps = math.ceil(
        settings.max_train_tokens
        / (settings.batch_size * settings.sequence_length)
    )
    if settings.warmup_steps > training_steps:
        raise ValueError(
            "warmup_steps cannot exceed the configured training steps; "
            f"got {settings.warmup_steps} > {training_steps}"
        )

    initial_token_budgets = allocate_dataset_token_budgets(
        settings.max_train_tokens,
        settings.dataset_token_percentages,
    )

    # Tokenize each allocated training prefix once before the first batch
    # reaches the model. The packed-batch iterators below only read token IDs,
    # so repeated epochs do not redo CPU tokenization work or scan unused
    # suffixes of a source dataset.
    train_splits = {
        dataset_name: pretokenize_split(
            source_split,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=initial_token_budgets[dataset_name],
        )
        for dataset_name, source_split in raw_train_sources.items()
    }
    validation_source = dataset.get("validation")
    validation_split = (
        pretokenize_split(
            validation_source,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.max_validation_tokens,
        )
        if validation_source is not None
        else None
    )

    _seed_everything(settings.seed)

    model_config_for_training = dict(model_config)
    model_config_for_training.setdefault("max_seq_len", settings.sequence_length)
    model = FraiseModel(model_config_for_training)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=settings.betas,
        weight_decay=settings.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=settings.warmup_steps,
        num_training_steps=training_steps,
    )
    pad_token_id = special_token_id(tokenizer, "<|pad|>")
    manifest = _run_manifest(
        dataset,
        tokenizer,
        model_config_for_training,
        settings,
    )
    save_manager = SaveManager(
        save_path,
        model_name,
        manifest=manifest,
        tokenizer=tokenizer,
        resume_from=resume_from,
        config_paths=config_paths,
    )

    state = TrainingState()
    if resume_from is not None:
        checkpoint = save_manager.load_checkpoint(model, optimizer, scheduler)
        state = TrainingState.from_checkpoint(checkpoint)
        save_manager.log(
            "Resuming from "
            f"{save_manager.latest_checkpoint} at global step "
            f"{state.global_step} and {state.tokens_seen} tokens."
        )
    else:
        save_manager.log(
            "Starting training: "
            f"{settings.max_train_tokens} target tokens, "
            f"{settings.sequence_length}-token context, "
            f"batch size {settings.batch_size}, "
            f"learning rate {settings.learning_rate}, "
            f"precision {'BF16' if settings.use_bf16 else 'FP32'}."
        )

    model_device = next(model.parameters()).device
    last_grad_norm: float | None = None

    for epoch in range(state.epoch, settings.num_epochs):
        if state.tokens_seen >= settings.max_train_tokens:
            break
        if epoch != state.epoch:
            state.start_epoch(epoch)

        start_batch_index = state.next_batch_index
        epoch_token_limit = state.epoch_tokens_seen + (
            settings.max_train_tokens - state.tokens_seen
        )
        epoch_token_budgets = allocate_dataset_token_budgets(
            epoch_token_limit,
            settings.dataset_token_percentages,
        )
        model.train()

        for batch_index, cpu_batch in iter_dataset_packed_batches(
            train_splits,
            epoch_token_budgets,
            settings.sequence_length,
            settings.packing_stride,
            pad_token_id,
            batch_size=settings.batch_size,
            start_batch_index=start_batch_index,
            seed=settings.seed + epoch,
            shuffle=settings.shuffle,
        ):
            batch = _move_batch(cpu_batch, model_device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(settings.use_bf16):
                result = _loss_for_batch(model, batch)
            if result is None:
                continue

            loss, valid_tokens = result
            # Cross-entropy is already averaged over the valid targets. A
            # short final batch should have a noisier gradient, not an
            # arbitrarily rescaled one.
            loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=settings.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(grad_norm_tensor).item()):
                raise FloatingPointError("non-finite gradient norm detected")

            grad_norm = float(grad_norm_tensor.detach().cpu())
            optimizer.step()
            scheduler.step()
            state.record_step(
                batch_index,
                float(loss.item()),
                valid_tokens,
            )
            last_grad_norm = grad_norm

            if state.global_step % settings.checkpoint_interval_steps == 0:
                save_manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    training_state=state.as_checkpoint(),
                )
                save_manager.log(
                    f"Checkpoint saved at global step {state.global_step} "
                    f"and {state.tokens_seen} tokens."
                    f" Last grad norm {last_grad_norm:.6f}"
                    f" learning rate {scheduler.get_last_lr()[0]:.8g}"
                    f" training loss {float(loss.item()):.6f}"
                )
                

            if state.tokens_seen >= settings.max_train_tokens:
                break

        if state.optimizer_steps_in_epoch == 0:
            raise RuntimeError(
                f"epoch {epoch + 1} produced no valid packed training batches"
            )

        train_loss = state.epoch_loss_sum / state.epoch_valid_tokens
        validation_loss = None
        if validation_split is not None:
            model.eval()
            validation_loss = _evaluate(
                model,
                validation_split,
                settings,
                pad_token_id,
            )

        optimizer_steps_in_epoch = state.optimizer_steps_in_epoch
        state.finish_epoch()
        save_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            training_state=state.as_checkpoint(),
        )

        validation_message = (
            f", validation loss {validation_loss:.6f}"
            if validation_loss is not None
            else ", validation loss unavailable"
        )
        current_lr = scheduler.get_last_lr()[0]
        save_manager.log(
            f"Epoch {epoch + 1}/{settings.num_epochs}: "
            f"training loss {train_loss:.6f}{validation_message}, "
            f"optimizer steps {optimizer_steps_in_epoch}, "
            f"tokens {state.tokens_seen}/{settings.max_train_tokens}, "
            f"learning rate {current_lr:.8g}, "
            f"last grad norm {last_grad_norm:.6f}"
        )

        if state.tokens_seen >= settings.max_train_tokens:
            break

    if state.tokens_seen < settings.max_train_tokens:
        raise RuntimeError(
            "training ended before reaching max_train_tokens: "
            f"{state.tokens_seen} / {settings.max_train_tokens}; "
            "increase num_epochs or provide more training data"
        )

    save_manager.finish(model, training_state=state.as_checkpoint())
    return model

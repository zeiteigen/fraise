"""The Fraise pretraining loop for the project's concrete model and data."""

from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
import math
import multiprocessing
from pathlib import Path
import random

import torch
from datasets import DatasetDict
from tokenizers import Tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from components import device
from components.model import FraiseModel, ModelConfig
from datasets_mgr import get_dataset_sources

from .batch import _autocast_context, _loss_for_batch, _move_batch
from .budgets import _plan_training_budgets
from .hyperparameters import (
    TrainingHyperparameters,
    _validate_hyperparameters,
)
from .manifest import _run_manifest
from .packing import iter_dataset_packed_batches, pretokenize_split, special_token_id
from .save_mgr import SaveManager
from .state import TrainingState
from .validation import _evaluate_checkpoint, _init_validation_worker


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

    # Tokenize each allocated training prefix once before the first batch
    # reaches the model. The packed-batch iterators below only read token IDs,
    # so repeated epochs do not redo CPU tokenization work or scan unused
    # suffixes of a source dataset.
    train_splits = {
        dataset_name: pretokenize_split(
            source_split,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.dataset_token_counts[dataset_name],
        )
        for dataset_name, source_split in raw_train_sources.items()
    }
    epoch_token_budgets = _plan_training_budgets(train_splits, settings)
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
    # Module.compile() compiles calls while keeping the original module and
    # state-dict keys intact for checkpoint save and resume.
    model.compile()

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
            f"precision {'BF16' if settings.use_bf16 else 'FP32'}, "
            "torch.compile enabled."
        )
    save_manager.log(
        "Source target tokens: "
        + ", ".join(
            f"{name}={count:,}"
            for name, count in settings.dataset_token_counts.items()
        )
    )

    model_device = next(model.parameters()).device
    last_grad_norm: float | None = None
    validation_executor = (
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_validation_worker,
            initargs=(
                model_config_for_training,
                validation_split,
                settings,
                pad_token_id,
            ),
        )
        if validation_split is not None
        else None
    )
    pending_validations: list[tuple[int, Future[float | None]]] = []
    latest_validation: tuple[int, float] | None = None

    def report_validations(*, wait: bool = False) -> None:
        nonlocal latest_validation
        while pending_validations and (wait or pending_validations[0][1].done()):
            step, future = pending_validations.pop(0)
            try:
                validation_loss = future.result()
            except Exception as exc:
                save_manager.log(f"Validation failed at global step {step}: {exc}")
                continue
            if validation_loss is None:
                save_manager.log(f"Validation loss unavailable at global step {step}.")
            else:
                latest_validation = (step, validation_loss)
                save_manager.log(
                    f"Validation loss at global step {step}: "
                    f"{validation_loss:.6f}"
                )

    def queue_validation(checkpoint_path: Path) -> None:
        if validation_executor is not None:
            try:
                future = validation_executor.submit(
                    _evaluate_checkpoint,
                    str(checkpoint_path),
                )
            except Exception as exc:
                save_manager.log(
                    f"Could not queue validation at global step "
                    f"{state.global_step}: {exc}"
                )
            else:
                pending_validations.append((state.global_step, future))

    for epoch in range(state.epoch, settings.num_epochs):
        if state.tokens_seen >= settings.max_train_tokens:
            break
        if epoch != state.epoch:
            state.start_epoch(epoch)

        start_batch_index = state.next_batch_index
        model.train()

        for batch_index, cpu_batch in iter_dataset_packed_batches(
            train_splits,
            epoch_token_budgets[epoch],
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
                checkpoint_path = save_manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    training_state=state.as_checkpoint(),
                )
                queue_validation(checkpoint_path)
                save_manager.log(
                    f"Checkpoint saved at global step {state.global_step} "
                    f"and {state.tokens_seen} tokens."
                    f" Last grad norm {last_grad_norm:.6f}"
                    f" learning rate {scheduler.get_last_lr()[0]:.8g}"
                    f" training loss {float(loss.item()):.6f}"
                )

            report_validations()

            if state.tokens_seen >= settings.max_train_tokens:
                break

        if state.optimizer_steps_in_epoch == 0:
            raise RuntimeError(
                f"epoch {epoch + 1} produced no valid packed training batches"
            )

        train_loss = state.epoch_loss_sum / state.epoch_valid_tokens
        optimizer_steps_in_epoch = state.optimizer_steps_in_epoch
        state.finish_epoch()
        checkpoint_path = save_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            training_state=state.as_checkpoint(),
        )
        if state.global_step % settings.checkpoint_interval_steps != 0:
            queue_validation(checkpoint_path)
        report_validations()

        if latest_validation is not None:
            validation_message = (
                f", latest validation loss {latest_validation[1]:.6f} "
                f"(step {latest_validation[0]})"
            )
        elif validation_split is not None:
            validation_message = ", validation pending"
        else:
            validation_message = ", validation loss unavailable"
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

    report_validations(wait=True)
    if validation_executor is not None:
        validation_executor.shutdown()
    save_manager.finish(model, training_state=state.as_checkpoint())
    return model

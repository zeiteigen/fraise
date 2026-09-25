"""Packed, reproducible, mixed-precision training utilities for Fraise."""

from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import platform
import random
from typing import Any, TypedDict

import torch
import torch.nn.functional as F
from datasets import DatasetDict
from tokenizers import Tokenizer
from torch import nn
from transformers.optimization import get_cosine_schedule_with_warmup

from components import device
from datasets_mgr import (
    get_dataset_fingerprint,
    get_dataset_name,
    get_dataset_size_bytes,
)
from helpers.get_git_commit import get_git_snapshot
from tokenizer import get_tokenizer_fingerprint
from train.save_mgr import SaveManager


class TrainingHyperparameters(TypedDict):
    """Configuration accepted by :func:`pretrain_transformer`."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_steps: int
    sequence_length: int
    packing_stride: int
    max_train_tokens: int
    max_validation_tokens: int | None
    seed: int
    checkpoint_interval_steps: int
    tokenization_batch_size: int
    use_bf16: bool
    shuffle: bool
    gradient_clip_norm: float


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_steps: int
    sequence_length: int
    packing_stride: int
    max_train_tokens: int
    max_validation_tokens: int | None
    seed: int
    checkpoint_interval_steps: int
    tokenization_batch_size: int
    use_bf16: bool
    shuffle: bool
    gradient_clip_norm: float


@dataclass
class PackedTokenBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    padding_mask: torch.Tensor | None
    valid_tokens: int


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _validate_hyperparameters(
    hyperparameters: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> TrainingSettings:
    """Validate and normalize all settings that affect training state."""
    batch_size = _positive_int("batch_size", hyperparameters.get("batch_size"))
    num_epochs = _positive_int("num_epochs", hyperparameters.get("num_epochs"))
    sequence_length = _positive_int(
        "sequence_length",
        hyperparameters.get(
            "sequence_length",
            model_config.get("max_seq_len", 512),
        ),
    )
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    packing_stride = _positive_int(
        "packing_stride",
        hyperparameters.get("packing_stride", sequence_length),
    )
    if packing_stride > sequence_length:
        raise ValueError(
            "packing_stride cannot exceed sequence_length; "
            f"got {packing_stride} > {sequence_length}"
        )

    max_train_tokens = _positive_int(
        "max_train_tokens",
        hyperparameters.get("max_train_tokens"),
    )

    max_validation_tokens = hyperparameters.get("max_validation_tokens")
    if max_validation_tokens is not None:
        max_validation_tokens = _positive_int(
            "max_validation_tokens",
            max_validation_tokens,
        )

    warmup_steps = hyperparameters.get("warmup_steps", 0)
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, Integral):
        raise ValueError(
            f"warmup_steps must be a non-negative integer; got {warmup_steps!r}"
        )
    if warmup_steps < 0:
        raise ValueError(
            f"warmup_steps must be a non-negative integer; got {warmup_steps!r}"
        )

    learning_rate = hyperparameters.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError(
            f"learning_rate must be a finite positive number; got {learning_rate!r}"
        )

    weight_decay = hyperparameters.get("weight_decay", 0.0)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, Real)
        or not math.isfinite(float(weight_decay))
        or weight_decay < 0
    ):
        raise ValueError(
            f"weight_decay must be a finite non-negative number; got {weight_decay!r}"
        )

    seed = hyperparameters.get("seed", 1337)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")

    checkpoint_interval_steps = _positive_int(
        "checkpoint_interval_steps",
        hyperparameters.get("checkpoint_interval_steps", 1000),
    )
    tokenization_batch_size = _positive_int(
        "tokenization_batch_size",
        hyperparameters.get("tokenization_batch_size", 1000),
    )

    use_bf16 = hyperparameters.get("use_bf16", False)
    if not isinstance(use_bf16, bool):
        raise ValueError(f"use_bf16 must be a bool; got {use_bf16!r}")
    shuffle = hyperparameters.get("shuffle", True)
    if not isinstance(shuffle, bool):
        raise ValueError(f"shuffle must be a bool; got {shuffle!r}")

    gradient_clip_norm = hyperparameters.get("gradient_clip_norm", 1.0)
    if (
        isinstance(gradient_clip_norm, bool)
        or not isinstance(gradient_clip_norm, Real)
        or not math.isfinite(float(gradient_clip_norm))
        or gradient_clip_norm <= 0
    ):
        raise ValueError(
            "gradient_clip_norm must be a finite positive number; "
            f"got {gradient_clip_norm!r}"
        )

    configured_max_seq_len = model_config.get("max_seq_len")
    if configured_max_seq_len is not None:
        configured_max_seq_len = _positive_int(
            "max_seq_len",
            configured_max_seq_len,
        )
        if configured_max_seq_len < sequence_length:
            raise ValueError(
                "sequence_length cannot exceed model_config['max_seq_len']; "
                f"got {sequence_length} > {configured_max_seq_len}"
            )

    return TrainingSettings(
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        num_epochs=num_epochs,
        warmup_steps=int(warmup_steps),
        sequence_length=sequence_length,
        packing_stride=packing_stride,
        max_train_tokens=max_train_tokens,
        max_validation_tokens=max_validation_tokens,
        seed=int(seed),
        checkpoint_interval_steps=checkpoint_interval_steps,
        tokenization_batch_size=tokenization_batch_size,
        use_bf16=use_bf16,
        shuffle=shuffle,
        gradient_clip_norm=float(gradient_clip_norm),
    )


def _iter_batches(split: Any, batch_size: int) -> Iterator[Mapping[str, Any]]:
    """Yield dictionary batches from a Hugging Face or compatible split."""
    batch_iterator = getattr(split, "iter", None)
    if callable(batch_iterator):
        yield from batch_iterator(batch_size=batch_size)
        return

    rows: list[Mapping[str, Any]] = []
    for row in split:
        if not isinstance(row, Mapping):
            raise TypeError(
                "dataset rows must be mappings containing a 'text' field; "
                f"got {type(row).__name__}"
            )
        rows.append(row)
        if len(rows) == batch_size:
            yield {key: [item[key] for item in rows] for key in rows[0]}
            rows = []
    if rows:
        yield {key: [item[key] for item in rows] for key in rows[0]}


def _batch_texts(batch: Mapping[str, Any]) -> list[str]:
    """Extract non-empty text strings from one dataset batch."""
    if "text" not in batch:
        raise ValueError(
            "training batches must contain a 'text' column; "
            f"available columns: {list(batch)}"
        )

    texts = batch["text"]
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, (list, tuple)):
        raise TypeError(
            "the 'text' column must contain strings or a batch of strings; "
            f"got {type(texts).__name__}"
        )
    return [text for text in texts if isinstance(text, str) and text]


def _encoding_ids(encoding: Any) -> list[int]:
    """Normalize a tokenizers ``Encoding`` or a plain ID sequence."""
    if hasattr(encoding, "ids"):
        encoding = encoding.ids
    if isinstance(encoding, torch.Tensor):
        encoding = encoding.detach().cpu().tolist()
    if not isinstance(encoding, (list, tuple)):
        raise TypeError(
            "tokenizer encodings must expose an 'ids' sequence; "
            f"got {type(encoding).__name__}"
        )
    return [int(token_id) for token_id in encoding]


def _encode_texts(
    tokenizer: Tokenizer,
    texts: list[str],
    sequence_length: int | None = None,
) -> list[list[int]]:
    """Encode texts while preserving tokenizer-defined BOS/EOS processing."""
    encode_batch = getattr(tokenizer, "encode_batch", None)
    if callable(encode_batch):
        encodings = encode_batch(texts, add_special_tokens=True)
        sequences = [_encoding_ids(encoding) for encoding in encodings]
        return (
            [sequence[:sequence_length] for sequence in sequences]
            if sequence_length is not None
            else sequences
        )

    tokenizer_call = getattr(tokenizer, "__call__", None)
    if not callable(tokenizer_call):
        raise TypeError(
            "tokenizer must provide encode_batch or a callable Transformers-style API"
        )

    call_kwargs = {"truncation": sequence_length is not None, "padding": False}
    if sequence_length is not None:
        call_kwargs["max_length"] = sequence_length
    encoded = tokenizer(texts, **call_kwargs)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer output must contain 'input_ids'")

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().cpu().tolist()
    if input_ids and isinstance(input_ids[0], Integral):
        input_ids = [input_ids]

    sequences = []
    for index, ids in enumerate(input_ids):
        ids = _encoding_ids(ids)
        if attention_mask is not None:
            mask = attention_mask[index]
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().tolist()
            ids = [token_id for token_id, is_valid in zip(ids, mask) if is_valid]
        sequences.append(ids[:sequence_length] if sequence_length else ids)
    return sequences


def _pad_token_id(tokenizer: Tokenizer) -> int:
    """Find the tokenizer's padding ID without changing tokenizer state."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, Integral) and not isinstance(pad_token_id, bool):
        return int(pad_token_id)

    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id):
        pad_token_id = token_to_id("<|pad|>")
        if pad_token_id is not None:
            return int(pad_token_id)

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        if "<|pad|>" in vocabulary:
            return int(vocabulary["<|pad|>"])

    raise ValueError(
        "tokenizer must define a padding token ID (expected '<|pad|>')"
    )


def _shuffled_split(split: Any, seed: int, shuffle: bool) -> Any:
    if not shuffle:
        return split
    shuffle_method = getattr(split, "shuffle", None)
    if callable(shuffle_method):
        return shuffle_method(seed=seed)

    rows = list(split)
    random.Random(seed).shuffle(rows)
    return rows


def _make_packed_block(
    token_ids: list[int],
    sequence_length: int,
    pad_token_id: int,
    valid_targets: int,
) -> PackedTokenBatch:
    if not 1 <= valid_targets <= sequence_length:
        raise ValueError(f"valid_targets must be in [1, {sequence_length}]")
    if len(token_ids) < valid_targets + 1:
        raise ValueError("a packed block needs one input token per target plus one")

    block = token_ids[: valid_targets + 1]
    block.extend([pad_token_id] * (sequence_length + 1 - len(block)))
    input_ids = torch.tensor(block[:-1], dtype=torch.long)
    targets = torch.tensor(block[1:], dtype=torch.long)

    target_mask = torch.zeros(sequence_length, dtype=torch.bool)
    target_mask[:valid_targets] = True
    padding_mask = None
    if valid_targets < sequence_length:
        padding_mask = torch.arange(sequence_length) >= valid_targets

    return PackedTokenBatch(
        input_ids=input_ids,
        targets=targets,
        target_mask=target_mask,
        padding_mask=padding_mask,
        valid_tokens=valid_targets,
    )


def _iter_packed_blocks(
    split: Any,
    tokenizer: Tokenizer,
    sequence_length: int,
    packing_stride: int,
    pad_token_id: int,
    *,
    tokenization_batch_size: int,
    seed: int,
    shuffle: bool,
    max_tokens: int | None,
) -> Iterator[PackedTokenBatch]:
    """Pack document encodings into fixed-context causal-LM windows.

    A stride equal to ``sequence_length`` gives non-overlapping packing. A
    smaller stride gives overlapping sliding windows. BOS/EOS tokens supplied
    by the tokenizer remain in the stream, so document boundaries are explicit.
    """
    split = _shuffled_split(split, seed, shuffle)
    buffer: list[int] = []
    produced_tokens = 0

    for batch in _iter_batches(split, tokenization_batch_size):
        texts = _batch_texts(batch)
        if not texts:
            continue
        for sequence in _encode_texts(tokenizer, texts):
            if not sequence:
                continue
            buffer.extend(sequence)
            while len(buffer) >= sequence_length + 1:
                remaining = (
                    max_tokens - produced_tokens
                    if max_tokens is not None
                    else sequence_length
                )
                if remaining <= 0:
                    return
                valid_targets = min(sequence_length, remaining)
                yield _make_packed_block(
                    buffer,
                    sequence_length,
                    pad_token_id,
                    valid_targets,
                )
                produced_tokens += valid_targets
                if valid_targets < sequence_length:
                    return
                del buffer[:packing_stride]

    if max_tokens is not None and produced_tokens >= max_tokens:
        return
    if len(buffer) < 2:
        return

    remaining = (
        max_tokens - produced_tokens if max_tokens is not None else sequence_length
    )
    valid_targets = min(sequence_length, len(buffer) - 1, remaining)
    if valid_targets > 0:
        yield _make_packed_block(
            buffer,
            sequence_length,
            pad_token_id,
            valid_targets,
        )


def _stack_packed_blocks(blocks: list[PackedTokenBatch]) -> PackedTokenBatch:
    padding_masks = [
        block.padding_mask
        if block.padding_mask is not None
        else torch.zeros_like(block.target_mask)
        for block in blocks
    ]
    has_padding = any(block.padding_mask is not None for block in blocks)
    return PackedTokenBatch(
        input_ids=torch.stack([block.input_ids for block in blocks]),
        targets=torch.stack([block.targets for block in blocks]),
        target_mask=torch.stack([block.target_mask for block in blocks]),
        padding_mask=torch.stack(padding_masks) if has_padding else None,
        valid_tokens=sum(block.valid_tokens for block in blocks),
    )


def _iter_packed_batches(
    split: Any,
    tokenizer: Tokenizer,
    sequence_length: int,
    packing_stride: int,
    pad_token_id: int,
    *,
    batch_size: int,
    tokenization_batch_size: int,
    seed: int,
    shuffle: bool,
    max_tokens: int | None,
    start_batch_index: int = 0,
) -> Iterator[tuple[int, PackedTokenBatch]]:
    blocks: list[PackedTokenBatch] = []
    batch_index = 0
    for block in _iter_packed_blocks(
        split,
        tokenizer,
        sequence_length,
        packing_stride,
        pad_token_id,
        tokenization_batch_size=tokenization_batch_size,
        seed=seed,
        shuffle=shuffle,
        max_tokens=max_tokens,
    ):
        blocks.append(block)
        if len(blocks) < batch_size:
            continue
        if batch_index >= start_batch_index:
            yield batch_index, _stack_packed_blocks(blocks)
        batch_index += 1
        blocks = []

    if blocks:
        if batch_index >= start_batch_index:
            yield batch_index, _stack_packed_blocks(blocks)


def _loss_for_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int,
    *,
    target_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int] | None:
    """Return finite mean loss and valid target count for one packed batch."""
    if padding_mask is None:
        logits = model(input_ids)
    else:
        logits = model(input_ids, padding_mask=padding_mask)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError(
            "the model must return logits shaped [batch, sequence, vocabulary]"
        )
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            "model logits and input IDs disagree on batch/sequence dimensions: "
            f"{tuple(logits.shape[:2])} vs {tuple(input_ids.shape)}"
        )

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if target_mask is None:
        valid_targets = flat_targets.ne(pad_token_id)
    else:
        if target_mask.shape != targets.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be a bool tensor shaped like targets")
        valid_targets = target_mask.reshape(-1).to(device=targets.device)

    valid_count = int(valid_targets.sum().item())
    if valid_count == 0:
        return None

    selected_targets = flat_targets[valid_targets]
    if int(selected_targets.min().item()) < 0 or int(
        selected_targets.max().item()
    ) >= logits.size(-1):
        raise ValueError("target token ID is outside the model vocabulary")

    loss = F.cross_entropy(flat_logits[valid_targets], selected_targets)
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
        padding_mask=(
            batch.padding_mask.to(target_device, non_blocking=True)
            if batch.padding_mask is not None
            else None
        ),
        valid_tokens=batch.valid_tokens,
    )


def _evaluate(
    model: nn.Module,
    split: Any,
    tokenizer: Tokenizer,
    settings: TrainingSettings,
    pad_token_id: int,
) -> float | None:
    """Evaluate packed validation tokens with token-weighted cross-entropy."""
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device
    max_tokens = settings.max_validation_tokens
    with torch.no_grad():
        for _, cpu_batch in _iter_packed_batches(
            split,
            tokenizer,
            settings.sequence_length,
            settings.packing_stride,
            pad_token_id,
            batch_size=settings.batch_size,
            tokenization_batch_size=settings.tokenization_batch_size,
            seed=settings.seed,
            shuffle=False,
            max_tokens=max_tokens,
        ):
            batch = _move_batch(cpu_batch, model_device)
            with _autocast_context(settings.use_bf16):
                result = _loss_for_batch(
                    model,
                    batch.input_ids,
                    batch.targets,
                    pad_token_id,
                    target_mask=batch.target_mask,
                    padding_mask=batch.padding_mask,
                )
            if result is None:
                continue
            loss, valid_count = result
            total_loss += float(loss.item()) * valid_count
            total_tokens += valid_count

    return total_loss / total_tokens if total_tokens else None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)


def _dataset_manifest(dataset: DatasetDict) -> dict[str, object]:
    splits: dict[str, object] = {}
    for split_name, split_dataset in sorted(dataset.items()):
        try:
            row_count = len(split_dataset)
        except TypeError:
            row_count = None
        splits[split_name] = {
            "rows": row_count,
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
    model_config: Mapping[str, Any],
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
            "pad_token_id": _pad_token_id(tokenizer),
            "bos_token_id": tokenizer.token_to_id("<|bos|>"),
            "eos_token_id": tokenizer.token_to_id("<|eos|>"),
        },
    }


def _state_from_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, Mapping):
        raise ValueError("checkpoint is missing training_state")
    defaults: dict[str, object] = {
        "epoch": 0,
        "next_batch_index": 0,
        "global_step": 0,
        "tokens_seen": 0,
        "epoch_tokens_seen": 0,
        "epoch_loss_sum": 0.0,
        "epoch_valid_tokens": 0,
        "optimizer_steps_in_epoch": 0,
    }
    defaults.update(training_state)
    for key in (
        "epoch",
        "next_batch_index",
        "global_step",
        "tokens_seen",
        "epoch_tokens_seen",
        "epoch_valid_tokens",
        "optimizer_steps_in_epoch",
    ):
        defaults[key] = int(defaults[key])
    defaults["epoch_loss_sum"] = float(defaults["epoch_loss_sum"])
    return defaults


def _checkpoint_state(
    state: Mapping[str, object],
    *,
    last_grad_norm: float | None = None,
) -> dict[str, object]:
    result = dict(state)
    if last_grad_norm is not None:
        result["last_grad_norm"] = last_grad_norm
    return result


def pretrain_transformer(
    transformer_model_class: type[nn.Module],
    tokenizer: Tokenizer,
    dataset: DatasetDict,
    hyperparameters: TrainingHyperparameters,
    model_config: dict,
    save_path: str,
    model_name: str = "fraise_model",
    resume_from: str | None = None,
) -> nn.Module:
    """Train a packed causal language model with resumable checkpoints.

    ``max_train_tokens`` is the hard target-token budget for the whole run.
    Each document is encoded with BOS/EOS, concatenated into a token stream,
    and emitted as fixed ``sequence_length`` target windows. ``packing_stride``
    controls overlap; equal to the context length means non-overlapping blocks.
    """
    if "train" not in dataset:
        raise ValueError("dataset must contain a 'train' split")

    settings = _validate_hyperparameters(hyperparameters, model_config)
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

    _seed_everything(settings.seed)

    model_config_for_training = dict(model_config)
    model_config_for_training.setdefault("max_seq_len", settings.sequence_length)
    model = transformer_model_class(model_config_for_training).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=settings.warmup_steps,
        num_training_steps=training_steps,
    )
    pad_token_id = _pad_token_id(tokenizer)
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
        resume_from=resume_from,
    )

    state: dict[str, object] = {
        "epoch": 0,
        "next_batch_index": 0,
        "global_step": 0,
        "tokens_seen": 0,
        "epoch_tokens_seen": 0,
        "epoch_loss_sum": 0.0,
        "epoch_valid_tokens": 0,
        "optimizer_steps_in_epoch": 0,
    }
    if resume_from is not None:
        checkpoint = save_manager.load_checkpoint(model, optimizer, scheduler)
        state = _state_from_checkpoint(checkpoint)
        save_manager.log(
            "Resuming from "
            f"{save_manager.latest_checkpoint} at global step "
            f"{state['global_step']} and {state['tokens_seen']} tokens."
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
    train_split = dataset["train"]
    validation_split = dataset.get("validation")
    last_grad_norm: float | None = None

    for epoch in range(int(state["epoch"]), settings.num_epochs):
        if int(state["tokens_seen"]) >= settings.max_train_tokens:
            break

        if epoch != int(state["epoch"]):
            state.update(
                {
                    "next_batch_index": 0,
                    "epoch_tokens_seen": 0,
                    "epoch_loss_sum": 0.0,
                    "epoch_valid_tokens": 0,
                    "optimizer_steps_in_epoch": 0,
                }
            )

        start_batch_index = int(state["next_batch_index"])
        epoch_tokens_seen = int(state["epoch_tokens_seen"])
        epoch_token_limit = epoch_tokens_seen + (
            settings.max_train_tokens - int(state["tokens_seen"])
        )
        model.train()

        for batch_index, cpu_batch in _iter_packed_batches(
            train_split,
            tokenizer,
            settings.sequence_length,
            settings.packing_stride,
            pad_token_id,
            batch_size=settings.batch_size,
            tokenization_batch_size=settings.tokenization_batch_size,
            seed=settings.seed + epoch,
            shuffle=settings.shuffle,
            max_tokens=epoch_token_limit,
            start_batch_index=start_batch_index,
        ):
            batch = _move_batch(cpu_batch, model_device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(settings.use_bf16):
                result = _loss_for_batch(
                    model,
                    batch.input_ids,
                    batch.targets,
                    pad_token_id,
                    target_mask=batch.target_mask,
                    padding_mask=batch.padding_mask,
                )
                if result is None:
                    continue
                loss, valid_count = result
                # Keep a short final batch from receiving a larger update than
                # a full batch merely because its mean loss uses fewer tokens.
                nominal_tokens = batch.input_ids.numel()
                loss_for_backward = loss * (valid_count / nominal_tokens)
            if not bool(torch.isfinite(loss_for_backward).item()):
                raise FloatingPointError("non-finite scaled loss detected")

            loss_for_backward.backward()
            last_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=settings.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(last_grad_norm_tensor).item()):
                raise FloatingPointError("non-finite gradient norm detected")
            last_grad_norm = float(last_grad_norm_tensor.detach().cpu())

            optimizer.step()
            scheduler.step()

            state["global_step"] = int(state["global_step"]) + 1
            state["tokens_seen"] = int(state["tokens_seen"]) + valid_count
            state["epoch_tokens_seen"] = int(state["epoch_tokens_seen"]) + valid_count
            state["next_batch_index"] = batch_index + 1
            state["epoch_loss_sum"] = (
                float(state["epoch_loss_sum"]) + float(loss.item()) * valid_count
            )
            state["epoch_valid_tokens"] = int(state["epoch_valid_tokens"]) + valid_count
            state["optimizer_steps_in_epoch"] = (
                int(state["optimizer_steps_in_epoch"]) + 1
            )

            if int(state["global_step"]) % settings.checkpoint_interval_steps == 0:
                save_manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    next_batch_index=int(state["next_batch_index"]),
                    global_step=int(state["global_step"]),
                    tokens_seen=int(state["tokens_seen"]),
                    epoch_tokens_seen=int(state["epoch_tokens_seen"]),
                    extra_state=_checkpoint_state(
                        state,
                        last_grad_norm=last_grad_norm,
                    ),
                )

            if int(state["tokens_seen"]) >= settings.max_train_tokens:
                break

        optimizer_steps_in_epoch = int(state["optimizer_steps_in_epoch"])
        if optimizer_steps_in_epoch == 0:
            raise RuntimeError(
                f"epoch {epoch + 1} produced no valid packed training batches"
            )

        train_loss = float(state["epoch_loss_sum"]) / int(
            state["epoch_valid_tokens"]
        )
        validation_loss = None
        if validation_split is not None:
            model.eval()
            validation_loss = _evaluate(
                model,
                validation_split,
                tokenizer,
                settings,
                pad_token_id,
            )

        completed_state = dict(state)
        completed_state.update(
            {
                "epoch": epoch + 1,
                "next_batch_index": 0,
                "epoch_tokens_seen": 0,
                "epoch_loss_sum": 0.0,
                "epoch_valid_tokens": 0,
                "optimizer_steps_in_epoch": 0,
            }
        )
        state = completed_state
        save_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=epoch + 1,
            next_batch_index=0,
            global_step=int(state["global_step"]),
            tokens_seen=int(state["tokens_seen"]),
            epoch_tokens_seen=0,
            extra_state=_checkpoint_state(
                state,
                last_grad_norm=last_grad_norm,
            ),
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
            f"tokens {state['tokens_seen']}/{settings.max_train_tokens}, "
            f"learning rate {current_lr:.8g}, "
            f"last grad norm {last_grad_norm:.6f}"
        )

        if int(state["tokens_seen"]) >= settings.max_train_tokens:
            break

    if int(state["tokens_seen"]) < settings.max_train_tokens:
        raise RuntimeError(
            "training ended before reaching max_train_tokens: "
            f"{state['tokens_seen']} / {settings.max_train_tokens}; "
            "increase num_epochs or provide more training data"
        )

    save_manager.finish(model, training_state=state)
    return model

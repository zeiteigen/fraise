"""Training hyperparameter definitions and validation."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, TypedDict


class TrainingHyperparameters(TypedDict):
    """Configuration accepted by :func:`pretrain_transformer`."""

    batch_size: int
    learning_rate: float
    betas: list[float]
    weight_decay: float
    dataset_token_counts: dict[str, int]
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
    betas: tuple[float, float]
    weight_decay: float
    dataset_token_counts: dict[str, int]
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


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _validate_dataset_token_counts(
    value: object,
    dataset_names: Collection[str] | None,
) -> dict[str, int]:
    """Validate absolute target-token counts for all training sources."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "dataset_token_counts must be a non-empty mapping of dataset names "
            f"to non-negative integers; got {value!r}"
        )

    counts: dict[str, int] = {}
    for dataset_name, count in value.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(
                "dataset_token_counts keys must be non-empty strings; "
                f"got {dataset_name!r}"
            )
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError(
                "dataset_token_counts values must be non-negative integers; "
                f"{dataset_name!r} got {count!r}"
            )
        counts[dataset_name] = int(count)

    if dataset_names is not None:
        available_names = set(dataset_names)
        configured_names = set(counts)
        unknown_names = sorted(configured_names - available_names)
        missing_names = sorted(available_names - configured_names)
        if unknown_names:
            raise ValueError(
                "dataset_token_counts contains unknown dataset name(s): "
                f"{', '.join(unknown_names)}. Available datasets: "
                f"{', '.join(sorted(available_names))}"
            )
        if missing_names:
            raise ValueError(
                "dataset_token_counts is missing dataset name(s): "
                f"{', '.join(missing_names)}"
            )
        return {name: counts[name] for name in sorted(available_names)}
    return counts


def _validate_hyperparameters(
    hyperparameters: Mapping[str, Any],
    model_config: Mapping[str, Any],
    dataset_names: Collection[str] | None = None,
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

    betas = hyperparameters.get("betas", (0.9, 0.999))
    if (
        isinstance(betas, (str, bytes))
        or not isinstance(betas, Sequence)
        or len(betas) != 2
    ):
        raise ValueError(
            "betas must be a sequence containing exactly two numbers; "
            f"got {betas!r}"
        )
    normalized_betas: list[float] = []
    for index, beta in enumerate(betas):
        if (
            isinstance(beta, bool)
            or not isinstance(beta, Real)
            or not math.isfinite(float(beta))
            or beta < 0
            or beta >= 1
        ):
            raise ValueError(
                f"betas[{index}] must be a finite number in [0, 1); got {beta!r}"
            )
        normalized_betas.append(float(beta))

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

    dataset_token_counts = _validate_dataset_token_counts(
        hyperparameters.get("dataset_token_counts"),
        dataset_names,
    )
    configured_total = sum(dataset_token_counts.values())
    if configured_total != max_train_tokens:
        raise ValueError(
            "dataset_token_counts must sum to max_train_tokens; "
            f"got {configured_total} != {max_train_tokens}"
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
        betas=(normalized_betas[0], normalized_betas[1]),
        weight_decay=float(weight_decay),
        dataset_token_counts=dataset_token_counts,
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

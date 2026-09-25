"""Training hyperparameter definitions and validation."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR
import math
from numbers import Integral, Real
from typing import Any, TypedDict


class TrainingHyperparameters(TypedDict):
    """Configuration accepted by :func:`pretrain_transformer`."""

    batch_size: int
    learning_rate: float
    betas: list[float]
    weight_decay: float
    dataset_token_percentages: dict[str, float]
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
    dataset_token_percentages: dict[str, float] = field(default_factory=dict)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _normalize_dataset_token_percentages(
    value: object,
    dataset_names: Collection[str] | None,
) -> dict[str, float]:
    """Validate and normalize the source-token percentage configuration."""
    if value is None:
        if dataset_names is None:
            return {}
        names = sorted(set(dataset_names))
        if not names:
            raise ValueError("at least one dataset is required for training")
        equal_share = 100.0 / len(names)
        percentages = {name: equal_share for name in names}
    else:
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                "dataset_token_percentages must be a non-empty mapping of "
                "dataset names to percentages; "
                f"got {value!r}"
            )

        percentages = {}
        for dataset_name, percentage in value.items():
            if not isinstance(dataset_name, str) or not dataset_name:
                raise ValueError(
                    "dataset_token_percentages keys must be non-empty strings; "
                    f"got {dataset_name!r}"
                )
            if (
                isinstance(percentage, bool)
                or not isinstance(percentage, Real)
                or not math.isfinite(float(percentage))
                or percentage < 0
                or percentage > 100
            ):
                raise ValueError(
                    "dataset_token_percentages values must be finite numbers "
                    "in [0, 100]; "
                    f"{dataset_name!r} got {percentage!r}"
                )
            percentages[dataset_name] = float(percentage)

        if dataset_names is not None:
            available_names = set(dataset_names)
            configured_names = set(percentages)
            unknown_names = sorted(configured_names - available_names)
            missing_names = sorted(available_names - configured_names)
            if unknown_names:
                raise ValueError(
                    "dataset_token_percentages contains unknown dataset name(s): "
                    f"{', '.join(unknown_names)}. Available datasets: "
                    f"{', '.join(sorted(available_names))}"
                )
            if missing_names:
                raise ValueError(
                    "dataset_token_percentages is missing dataset name(s): "
                    f"{', '.join(missing_names)}"
                )

    total_percentage = math.fsum(percentages.values())
    if not math.isclose(total_percentage, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "dataset_token_percentages must sum to 100; "
            f"got {total_percentage:g}"
        )

    # A tiny JSON floating-point rounding error should not turn into a token
    # allocation error. Normalize once, then retain deterministic key order.
    if total_percentage != 100.0:
        scale = 100.0 / total_percentage
        percentages = {
            name: percentage * scale
            for name, percentage in percentages.items()
        }

    if dataset_names is not None:
        return {
            name: percentages[name]
            for name in sorted(set(dataset_names))
        }
    return percentages


def allocate_dataset_token_budgets(
    total_tokens: int,
    dataset_token_percentages: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an exact integer token budget using largest remainders.

    Percentages usually do not map to whole tokens.  Flooring each share and
    assigning the remaining tokens to the largest fractional remainders keeps
    the sum exactly equal to ``total_tokens`` while minimizing rounding error.
    """
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, Integral):
        raise ValueError(f"total_tokens must be an integer; got {total_tokens!r}")
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative; got {total_tokens!r}")
    if not dataset_token_percentages:
        raise ValueError("at least one dataset token percentage is required")

    normalized = _normalize_dataset_token_percentages(
        dataset_token_percentages,
        dataset_names=None,
    )
    order = {dataset_name: index for index, dataset_name in enumerate(normalized)}
    raw_budgets = {
        name: Decimal(total_tokens) * Decimal(str(percentage)) / Decimal(100)
        for name, percentage in normalized.items()
    }
    budgets = {
        name: int(raw_budget.to_integral_value(rounding=ROUND_FLOOR))
        for name, raw_budget in raw_budgets.items()
    }
    remaining = int(total_tokens) - sum(budgets.values())
    remainders = sorted(
        normalized,
        key=lambda name: (
            -(raw_budgets[name] - budgets[name]),
            order[name],
        ),
    )
    for dataset_name in remainders[:remaining]:
        budgets[dataset_name] += 1
    return budgets


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

    dataset_token_percentages = _normalize_dataset_token_percentages(
        hyperparameters.get("dataset_token_percentages"),
        dataset_names,
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
        dataset_token_percentages=dataset_token_percentages,
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

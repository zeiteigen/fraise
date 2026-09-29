"""Normalized training settings; defaults and validation live here."""

from collections.abc import Collection
from dataclasses import dataclass

from .values import integer, number


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    learning_rate: float
    dataset_token_counts: dict[str, int]
    max_train_tokens: int
    num_epochs: int
    minibatch_size: int | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    warmup_steps: int = 0
    sequence_length: int = 512
    packing_stride: int | None = None
    max_validation_tokens: int | None = None
    seed: int = 1337
    checkpoint_interval_steps: int = 1000
    checkpoint_keep_last: int = 2
    tokenization_batch_size: int = 1000
    use_bf16: bool = False
    shuffle: bool = True
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        if self.minibatch_size is None:
            object.__setattr__(self, "minibatch_size", self.batch_size)
        if self.packing_stride is None:
            object.__setattr__(self, "packing_stride", self.sequence_length)
        for name in (
            "batch_size",
            "minibatch_size",
            "max_train_tokens",
            "num_epochs",
            "checkpoint_interval_steps",
            "checkpoint_keep_last",
            "tokenization_batch_size",
        ):
            integer(name, getattr(self, name))
        integer("sequence_length", self.sequence_length, 2)
        integer("packing_stride", self.packing_stride)
        integer("warmup_steps", self.warmup_steps, 0)
        integer("seed", self.seed, 0)
        if self.max_validation_tokens is not None:
            integer("max_validation_tokens", self.max_validation_tokens)
        if self.batch_size % self.minibatch_size:
            raise ValueError("minibatch_size must divide batch_size")
        if self.packing_stride > self.sequence_length:
            raise ValueError("packing_stride cannot exceed sequence_length")
        number("learning_rate", self.learning_rate)
        number("weight_decay", self.weight_decay, inclusive=True)
        number("gradient_clip_norm", self.gradient_clip_norm)
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("betas must contain exactly two numbers")
        for beta in self.betas:
            number("betas", beta, inclusive=True)
            if beta >= 1:
                raise ValueError("betas must be in [0, 1)")
        object.__setattr__(self, "betas", tuple(self.betas))
        for name in ("use_bf16", "shuffle"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if (
            not isinstance(self.dataset_token_counts, dict)
            or not self.dataset_token_counts
        ):
            raise ValueError("dataset_token_counts must be a non-empty dictionary")
        for name, count in self.dataset_token_counts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("dataset_token_counts keys must be non-empty strings")
            integer(f"dataset_token_counts[{name!r}]", count, 0)
        if sum(self.dataset_token_counts.values()) != self.max_train_tokens:
            raise ValueError("dataset_token_counts must sum to max_train_tokens")
        object.__setattr__(
            self, "dataset_token_counts", dict(self.dataset_token_counts)
        )

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.minibatch_size

    @classmethod
    def from_dict(cls, values: dict) -> "TrainingSettings":
        values = dict(values)
        has_accumulation = "gradient_accumulation_steps" in values
        accumulation = values.pop("gradient_accumulation_steps", None)
        if has_accumulation:
            integer("gradient_accumulation_steps", accumulation)
            integer("batch_size", values.get("batch_size"))
            if values["batch_size"] % accumulation:
                raise ValueError("gradient_accumulation_steps must divide batch_size")
            minibatch = values["batch_size"] // accumulation
            if "minibatch_size" in values and values["minibatch_size"] != minibatch:
                raise ValueError(
                    "minibatch_size * gradient_accumulation_steps must equal batch_size"
                )
            values["minibatch_size"] = minibatch
        return cls(**values)

    def validate_sources(self, names: Collection[str]) -> None:
        unknown = self.dataset_token_counts.keys() - set(names)
        missing = set(names) - self.dataset_token_counts.keys()
        if unknown:
            raise ValueError(f"unknown dataset name(s): {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"missing dataset name(s): {', '.join(sorted(missing))}")

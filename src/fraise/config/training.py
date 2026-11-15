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
    minibatch_size: int | None = None
    adamw_learning_rate: float = 0.0003
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_steps: int = 0
    sequence_length: int = 512
    packing_stride: int | None = None
    max_validation_tokens: int | None = 32768
    validation_timeout_seconds: float = 300.0
    validation_cpu_threads: int = 4
    validation_device: str = "cpu"
    seed: int = 1337
    checkpoint_interval_steps: int = 1000
    checkpoint_keep_last: int = 2
    tokenization_batch_size: int = 1000
    use_bf16: bool = False
    use_compile: bool = True
    compile_threads: int = 4
    deterministic: bool = False
    log_interval_steps: int = 100
    shuffle_data: bool = False
    gradient_clip_norm: float = 1.0
    tied_weight_init_std: float = 0.02
    # Untied input embeddings: None keeps PyTorch's default N(0, 1).
    embedding_init_std: float | None = None

    def __post_init__(self):
        if self.minibatch_size is None:
            object.__setattr__(self, "minibatch_size", self.batch_size)
        if self.packing_stride is None:
            object.__setattr__(self, "packing_stride", self.sequence_length)
        for name in (
            "batch_size",
            "minibatch_size",
            "max_train_tokens",
            "checkpoint_interval_steps",
            "checkpoint_keep_last",
            "tokenization_batch_size",
            "validation_cpu_threads",
            "muon_ns_steps",
            "compile_threads",
            "log_interval_steps",
        ):
            integer(name, getattr(self, name))
        integer("sequence_length", self.sequence_length, 2)
        integer("packing_stride", self.packing_stride)
        integer("warmup_steps", self.warmup_steps, 0)
        integer("seed", self.seed, 0)
        if self.max_validation_tokens is not None:
            integer("max_validation_tokens", self.max_validation_tokens, 0)
        if self.batch_size % self.minibatch_size:
            raise ValueError("minibatch_size must divide batch_size")
        if self.packing_stride > self.sequence_length:
            raise ValueError("packing_stride cannot exceed sequence_length")
        number("learning_rate", self.learning_rate)
        number("adamw_learning_rate", self.adamw_learning_rate)
        number("muon_momentum", self.muon_momentum, inclusive=True)
        if self.muon_momentum >= 1:
            raise ValueError("muon_momentum must be in [0, 1)")
        number("weight_decay", self.weight_decay, inclusive=True)
        number("gradient_clip_norm", self.gradient_clip_norm)
        number("tied_weight_init_std", self.tied_weight_init_std)
        if self.embedding_init_std is not None:
            number("embedding_init_std", self.embedding_init_std)
        number("validation_timeout_seconds", self.validation_timeout_seconds)
        if self.validation_device not in ("cpu", "npu"):
            raise ValueError("validation_device must be 'cpu' or 'npu'")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("betas must contain exactly two numbers")
        for beta in self.betas:
            number("betas", beta, inclusive=True)
            if beta >= 1:
                raise ValueError("betas must be in [0, 1)")
        object.__setattr__(self, "betas", tuple(self.betas))
        for name in (
            "use_bf16",
            "use_compile",
            "deterministic",
            "shuffle_data",
            "muon_nesterov",
        ):
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
    def from_dict(cls, values: dict) -> TrainingSettings:
        return cls(**values)

    def validate_sources(self, names: Collection[str]) -> None:
        unknown = self.dataset_token_counts.keys() - set(names)
        missing = set(names) - self.dataset_token_counts.keys()
        if unknown:
            raise ValueError(f"unknown dataset name(s): {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"missing dataset name(s): {', '.join(sorted(missing))}")

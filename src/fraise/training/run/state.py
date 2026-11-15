"""Checkpointable counters for a training run."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass
class TrainingState:
    """Counters needed to resume the single training pass mid-way.

    CheckpointStore stores and restores RNG state alongside these counters; RNG
    state is deliberately not part of this small bookkeeping object.
    """

    next_batch_index: int = 0
    global_step: int = 0
    # Budget units (base tokens for macro streams) trained so far.
    tokens_seen: int = 0
    loss_sum: float = 0.0
    # Loss-weighting targets; differs from tokens_seen only for macro streams.
    loss_targets: int = 0

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, object]) -> "TrainingState":
        saved = checkpoint.get("training_state")
        if not isinstance(saved, Mapping):
            raise ValueError("checkpoint is missing training_state")
        if "loss_sum" not in saved:
            raise ValueError(
                "checkpoint predates single-pass training; start a fresh run"
            )
        return cls(
            next_batch_index=int(saved["next_batch_index"]),
            global_step=int(saved["global_step"]),
            tokens_seen=int(saved["tokens_seen"]),
            loss_sum=float(saved["loss_sum"]),
            loss_targets=int(saved["loss_targets"]),
        )

    def record_step(
        self,
        batch_index: int,
        loss: float,
        valid_tokens: int,
        units: int | None = None,
    ) -> None:
        """Record a step's loss targets and its budget units (default: targets)."""
        units = valid_tokens if units is None else units
        self.global_step += 1
        self.tokens_seen += units
        self.next_batch_index = batch_index + 1
        self.loss_sum += loss * valid_tokens
        self.loss_targets += valid_tokens

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.loss_targets

    def as_checkpoint(self) -> dict[str, object]:
        return asdict(self)

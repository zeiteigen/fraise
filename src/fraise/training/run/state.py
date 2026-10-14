"""Checkpointable counters for a training run."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass
class TrainingState:
    """Counters needed to resume within an epoch.

    CheckpointStore stores and restores RNG state alongside these counters; RNG
    state is deliberately not part of this small bookkeeping object.
    """

    epoch: int = 0
    next_batch_index: int = 0
    global_step: int = 0
    tokens_seen: int = 0
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
            epoch_loss_sum=float(saved["epoch_loss_sum"]),
            epoch_valid_tokens=int(saved["epoch_valid_tokens"]),
            optimizer_steps_in_epoch=int(saved["optimizer_steps_in_epoch"]),
        )

    def record_step(
        self,
        batch_index: int,
        loss: float,
        valid_tokens: int,
    ) -> None:
        self.global_step += 1
        self.tokens_seen += valid_tokens
        self.next_batch_index = batch_index + 1
        self.epoch_loss_sum += loss * valid_tokens
        self.epoch_valid_tokens += valid_tokens
        self.optimizer_steps_in_epoch += 1

    def finish_epoch(self) -> None:
        self.epoch += 1
        self.next_batch_index = 0
        self.epoch_loss_sum = 0.0
        self.epoch_valid_tokens = 0
        self.optimizer_steps_in_epoch = 0

    def as_checkpoint(self) -> dict[str, object]:
        return asdict(self)

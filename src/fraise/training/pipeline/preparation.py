"""Prepare the exact tokenized sources and epoch plan shared by training and preview."""

from dataclasses import dataclass

import numpy as np
from datasets import Dataset

from fraise.data.types import TrainingData
from fraise.tokenizer.macro import unit_lengths as tokenizer_unit_lengths

from .budgets import plan_epochs, steps_for_windows
from .tokenization import pretokenize_split


@dataclass
class PreparedData:
    train: dict[str, Dataset]
    validation: Dataset | None
    budgets: list[dict[str, int]]
    steps: int
    windows: list[dict[str, int]] | None = None
    # Base tokens per ID for macro streams; None when each ID is one target unit.
    unit_lengths: np.ndarray | None = None

    @property
    def total_targets(self) -> int:
        return sum(sum(epoch.values()) for epoch in self.budgets)


def prepare_data(data: TrainingData, tokenizer, settings) -> PreparedData:
    active = {
        name for name, count in settings.dataset_token_counts.items() if count > 0
    }
    if active != data.train.keys():
        raise ValueError(
            "training sources do not match the positive dataset_token_counts"
        )
    train = {
        name: pretokenize_split(
            source,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.dataset_token_counts[name],
        )
        for name, source in data.train.items()
    }
    lengths = tokenizer_unit_lengths(tokenizer)
    budgets, windows = plan_epochs(train, settings, unit_lengths=lengths)
    steps = steps_for_windows(windows, batch_size=settings.batch_size)
    if settings.warmup_steps > steps:
        raise ValueError("warmup_steps cannot exceed planned training steps")
    validation = (
        pretokenize_split(
            data.validation,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.max_validation_tokens,
        )
        if data.validation is not None and settings.max_validation_tokens != 0
        else None
    )
    return PreparedData(train, validation, budgets, steps, windows, lengths)

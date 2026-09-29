"""Explicit source boundaries and validation data."""

from dataclasses import dataclass, field

from datasets import Dataset


@dataclass
class TrainingData:
    train: dict[str, Dataset] = field(default_factory=dict)
    validation: Dataset | None = None

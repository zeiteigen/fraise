"""Prepare the exact tokenized sources and epoch plan shared by training and preview."""

from dataclasses import dataclass

from datasets import Dataset

from fraise.data.types import TrainingData

from .budgets import _plan_training_budgets, _planned_training_steps
from .tokenization import pretokenize_split


@dataclass
class PreparedData:
    train: dict[str, Dataset]
    validation: Dataset | None
    budgets: list[dict[str, int]]
    steps: int


def prepare_data(
    data: TrainingData, tokenizer, settings, model_config=None
) -> PreparedData:
    active = {
        name for name, count in settings.dataset_token_counts.items() if count > 0
    }
    if active != data.train.keys():
        raise ValueError(
            "training sources do not match the positive dataset_token_counts"
        )
    if model_config is not None and settings.sequence_length > model_config.max_seq_len:
        raise ValueError("sequence_length cannot exceed model max_seq_len")
    train = {
        name: pretokenize_split(
            source,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.dataset_token_counts[name],
        )
        for name, source in data.train.items()
    }
    budgets = _plan_training_budgets(train, settings)
    steps = _planned_training_steps(
        budgets,
        sequence_length=settings.sequence_length,
        batch_size=settings.batch_size,
    )
    if settings.warmup_steps > steps:
        raise ValueError("warmup_steps cannot exceed planned training steps")
    validation = (
        pretokenize_split(
            data.validation,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.max_validation_tokens,
        )
        if data.validation is not None
        else None
    )
    return PreparedData(train, validation, budgets, steps)

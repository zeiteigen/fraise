"""Plan per-source token targets for each training epoch."""

from collections.abc import Mapping

from datasets import Dataset

from .hyperparameters import TrainingSettings
from .packing import packed_target_capacity


def _plan_training_budgets(
    train_splits: Mapping[str, Dataset],
    settings: TrainingSettings,
) -> list[dict[str, int]]:
    """Plan exact source counts across epochs, rejecting unavailable targets."""
    capacities: dict[str, int] = {}
    for dataset_name, token_budget in settings.dataset_token_counts.items():
        if token_budget == 0:
            capacities[dataset_name] = 0
            continue
        available_per_epoch = packed_target_capacity(
            train_splits[dataset_name],
            settings.sequence_length,
            settings.packing_stride,
        )
        capacities[dataset_name] = available_per_epoch
        if token_budget > available_per_epoch * settings.num_epochs:
            raise ValueError(
                f"dataset {dataset_name!r} is assigned {token_budget:,} of "
                f"{settings.max_train_tokens:,} target tokens "
                f"but its tokenized training split can supply at most "
                f"{available_per_epoch:,} target tokens per epoch "
                f"({available_per_epoch * settings.num_epochs:,} across "
                f"{settings.num_epochs} epoch(s)); reduce its count or "
                "increase num_epochs or provide more training data"
            )

    remaining = dict(settings.dataset_token_counts)
    plan: list[dict[str, int]] = []
    for _ in range(settings.num_epochs):
        budgets = {
            name: min(remaining[name], capacities[name])
            for name in remaining
        }
        plan.append(budgets)
        for name, budget in budgets.items():
            remaining[name] -= budget
    return plan


def _planned_training_steps(
    epoch_token_budgets: list[dict[str, int]],
    *,
    sequence_length: int,
    batch_size: int,
) -> int:
    """Count optimizer steps, including short source and epoch tails."""
    steps = 0
    for budgets in epoch_token_budgets:
        blocks = sum(
            (count + sequence_length - 1) // sequence_length
            for count in budgets.values()
        )
        steps += (blocks + batch_size - 1) // batch_size
    return steps

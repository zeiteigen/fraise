"""Plan per-source token targets for each training epoch."""

from collections.abc import Mapping

from datasets import Dataset

from fraise.config.training import TrainingSettings

from .packing import packed_target_capacity


def _plan_training_budgets(
    train_splits: Mapping[str, Dataset],
    settings: TrainingSettings,
) -> list[dict[str, int]]:
    """Repeat exact per-source counts each epoch, rejecting unavailable targets."""
    for dataset_name, token_budget in settings.dataset_token_counts.items():
        if token_budget == 0:
            continue
        available_per_epoch = packed_target_capacity(
            train_splits[dataset_name],
            settings.sequence_length,
            settings.packing_stride,
        )
        if token_budget > available_per_epoch:
            raise ValueError(
                f"dataset {dataset_name!r} is assigned {token_budget:,} of "
                f"{settings.max_train_tokens:,} target tokens per epoch "
                f"but its tokenized training split can supply at most "
                f"{available_per_epoch:,} target tokens per epoch; "
                "reduce its count or provide more training data"
            )

    return [dict(settings.dataset_token_counts) for _ in range(settings.num_epochs)]


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

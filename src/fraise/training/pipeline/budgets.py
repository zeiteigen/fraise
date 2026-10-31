"""Plan per-source token targets for each training epoch."""

from collections.abc import Mapping

from datasets import Dataset

from fraise.config.training import TrainingSettings

from .packing import packed_target_capacity, plan_unit_windows


def epoch_seed(settings: TrainingSettings, epoch: int) -> int:
    return settings.seed + epoch if settings.shuffle_per_epoch else settings.seed


def _capacity_error(dataset_name, token_budget, settings, available):
    return ValueError(
        f"dataset {dataset_name!r} is assigned {token_budget:,} of "
        f"{settings.max_train_tokens:,} target tokens per epoch "
        f"but its tokenized training split can supply at most "
        f"{available:,} target tokens per epoch; "
        "reduce its count or provide more training data"
    )


def plan_epochs(
    train_splits: Mapping[str, Dataset],
    settings: TrainingSettings,
    *,
    unit_lengths=None,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """Per-epoch exact target budgets and window counts for each active source.

    Base-token streams reproduce the configured counts. Macro streams count
    base-token units and finish each source with the macro that reaches its
    count, so their exact budgets can exceed it by less than one macro.
    """
    if unit_lengths is None:
        budgets = plan_training_budgets(train_splits, settings)
        windows = [
            {
                name: (count + settings.sequence_length - 1) // settings.sequence_length
                for name, count in epoch.items()
                if count > 0
            }
            for epoch in budgets
        ]
        return budgets, windows
    budgets, windows = [], []
    for epoch in range(settings.num_epochs):
        if epoch and not settings.shuffle_per_epoch:
            budgets.append(dict(budgets[0]))
            windows.append(dict(windows[0]))
            continue
        epoch_budgets, epoch_windows = {}, {}
        for name, token_budget in settings.dataset_token_counts.items():
            if token_budget == 0:
                epoch_budgets[name] = 0
                continue
            plan = plan_unit_windows(
                train_splits[name],
                settings.sequence_length,
                settings.packing_stride,
                token_budget,
                unit_lengths,
                seed=epoch_seed(settings, epoch),
                shuffle=settings.shuffle_per_epoch,
            )
            if plan.units < token_budget:
                raise _capacity_error(name, token_budget, settings, plan.units)
            epoch_budgets[name], epoch_windows[name] = plan.units, plan.windows
        budgets.append(epoch_budgets)
        windows.append(epoch_windows)
    return budgets, windows


def plan_training_budgets(
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
            raise _capacity_error(
                dataset_name, token_budget, settings, available_per_epoch
            )

    return [dict(settings.dataset_token_counts) for _ in range(settings.num_epochs)]


def planned_training_steps(
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


def steps_for_windows(epoch_window_counts: list[dict[str, int]], *, batch_size: int):
    return sum(
        (sum(counts.values()) + batch_size - 1) // batch_size
        for counts in epoch_window_counts
    )

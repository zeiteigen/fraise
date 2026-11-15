"""Plan each source's exact token targets for the single training pass."""

from collections.abc import Mapping

from datasets import Dataset

from fraise.config.training import TrainingSettings

from .packing import packed_target_capacity, plan_unit_windows


def _capacity_error(dataset_name, token_budget, settings, available):
    return ValueError(
        f"dataset {dataset_name!r} is assigned {token_budget:,} of "
        f"{settings.max_train_tokens:,} target tokens "
        f"but its tokenized training split can supply at most "
        f"{available:,} target tokens; "
        "reduce its count or provide more training data"
    )


def plan_training(
    train_splits: Mapping[str, Dataset],
    settings: TrainingSettings,
    *,
    unit_lengths=None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Exact target budgets and window counts for each active source.

    Base-token streams reproduce the configured counts. Macro streams count
    base-token units and finish each source with the macro that reaches its
    count, so their exact budgets can exceed it by less than one macro.
    """
    if unit_lengths is None:
        budgets = plan_training_budgets(train_splits, settings)
        windows = {
            name: (count + settings.sequence_length - 1) // settings.sequence_length
            for name, count in budgets.items()
            if count > 0
        }
        return budgets, windows
    budgets, windows = {}, {}
    for name, token_budget in settings.dataset_token_counts.items():
        if token_budget == 0:
            budgets[name] = 0
            continue
        plan = plan_unit_windows(
            train_splits[name],
            settings.sequence_length,
            settings.packing_stride,
            token_budget,
            unit_lengths,
        )
        if plan.units < token_budget:
            raise _capacity_error(name, token_budget, settings, plan.units)
        budgets[name], windows[name] = plan.units, plan.windows
    return budgets, windows


def plan_training_budgets(
    train_splits: Mapping[str, Dataset],
    settings: TrainingSettings,
) -> dict[str, int]:
    """The configured per-source counts, rejecting unavailable targets."""
    for dataset_name, token_budget in settings.dataset_token_counts.items():
        if token_budget == 0:
            continue
        available = packed_target_capacity(
            train_splits[dataset_name],
            settings.sequence_length,
            settings.packing_stride,
        )
        if token_budget > available:
            raise _capacity_error(dataset_name, token_budget, settings, available)
    return dict(settings.dataset_token_counts)


def planned_training_steps(
    token_budgets: Mapping[str, int],
    *,
    sequence_length: int,
    batch_size: int,
) -> int:
    """Count optimizer steps, including short source tails."""
    blocks = sum(
        (count + sequence_length - 1) // sequence_length
        for count in token_budgets.values()
    )
    return (blocks + batch_size - 1) // batch_size


def steps_for_windows(window_counts: Mapping[str, int], *, batch_size: int) -> int:
    return (sum(window_counts.values()) + batch_size - 1) // batch_size

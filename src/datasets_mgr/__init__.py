"""Utilities for composing the datasets used by Fraise."""

from collections.abc import Sequence
from typing import TypeAlias

from datasets import Dataset, IterableDataset, concatenate_datasets


DatasetSource: TypeAlias = Dataset | IterableDataset


def unified_dataset_loader(datasets: Sequence[DatasetSource]) -> DatasetSource:
    """Combine dataset sources into one dataset.

    Dataset-specific loaders such as :func:`use_babylm` and
    :func:`use_fineweb_edu` return a Hugging Face ``Dataset``.  Keeping the
    composition here means callers can train against one dataset regardless
    of how many sources are configured.

    Args:
        datasets: Non-empty sequence of compatible Hugging Face datasets.

    Returns:
        A dataset containing the rows from each input dataset in order.

    Raises:
        ValueError: If no dataset sources are provided.
    """
    if not datasets:
        raise ValueError("unified_dataset_loader requires at least one dataset")

    return concatenate_datasets(list(datasets))

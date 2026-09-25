"""Utilities for composing and fingerprinting the datasets used by Fraise."""

from collections.abc import Sequence
import hashlib
import json
from typing import TypeAlias

from datasets import (
    Dataset,
    DatasetDict,
    DatasetInfo,
    IterableDataset,
    concatenate_datasets,
)


DatasetSource: TypeAlias = Dataset | IterableDataset


allowed_splits: tuple[str, ...] = ("train", "validation", "test")


def _dataset_names(dataset: DatasetSource | DatasetDict) -> set[str]:
    """Return the non-empty dataset names associated with a source."""
    if not isinstance(dataset, (Dataset, IterableDataset, DatasetDict)):
        raise TypeError(
            "Expected datasets.Dataset, datasets.IterableDataset, or "
            "datasets.DatasetDict, "
            f"got {type(dataset).__name__}"
        )

    source_datasets = (
        dataset.values() if isinstance(dataset, DatasetDict) else (dataset,)
    )
    return {
        dataset_name
        for source_dataset in source_datasets
        if isinstance(
            dataset_name := getattr(
                getattr(source_dataset, "info", None),
                "dataset_name",
                None,
            ),
            str,
        )
        and dataset_name
    }


def get_dataset_name(dataset: DatasetSource | DatasetDict) -> str | None:
    """Return a stable name for a dataset or collection of dataset splits."""
    dataset_names = _dataset_names(dataset)
    return "+".join(sorted(dataset_names)) or None


def get_dataset_size_bytes(dataset: DatasetSource | DatasetDict) -> int | None:
    """Return the dataset size in bytes when it is available."""
    if not isinstance(dataset, (Dataset, IterableDataset, DatasetDict)):
        raise TypeError(
            "Expected datasets.Dataset, datasets.IterableDataset, or "
            "datasets.DatasetDict, "
            f"got {type(dataset).__name__}"
        )

    source_datasets = (
        list(dataset.values()) if isinstance(dataset, DatasetDict) else [dataset]
    )
    sizes: list[int] = []
    sizes_from_info: list[bool] = []

    for source_dataset in source_datasets:
        info = getattr(source_dataset, "info", None)
        size = getattr(info, "size_in_bytes", None)
        from_info = isinstance(size, int) and size >= 0

        if not from_info:
            size = getattr(info, "dataset_size", None)
            from_info = isinstance(size, int) and size >= 0

        if not from_info and isinstance(source_dataset, Dataset):
            size = int(source_dataset.data.nbytes)

        if not isinstance(size, int) or size < 0:
            return None

        sizes.append(size)
        sizes_from_info.append(from_info)

    if not sizes:
        return None

    # DatasetDict splits loaded by Hugging Face commonly share one
    # DatasetInfo whose size_in_bytes already covers every split.
    if all(sizes_from_info) and len(set(sizes)) == 1:
        return sizes[0]

    return sum(sizes)


def _dataset_fingerprint_payload(dataset: DatasetSource) -> dict[str, object]:
    """Return a stable, compact descriptor for one map-style dataset.

    Hugging Face datasets expose a content-derived fingerprint for datasets
    loaded from local files and for most dataset transformations.  Keeping the
    descriptor compact matters here: hashing every text byte at every launch
    would add a needless full-dataset pass before training starts.
    """
    try:
        num_rows = len(dataset)
    except TypeError:
        num_rows = None

    info = getattr(dataset, "info", None)
    return {
        "dataset_name": getattr(info, "dataset_name", None),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows": num_rows,
        "column_names": list(getattr(dataset, "column_names", ())),
        "features": str(getattr(dataset, "features", None)),
        "size_in_bytes": getattr(info, "size_in_bytes", None),
        "dataset_size": getattr(info, "dataset_size", None),
    }


def get_dataset_fingerprint(
    dataset: DatasetSource | DatasetDict,
    *,
    split: str | None = None,
) -> str:
    """Return a stable SHA-256 fingerprint for a dataset or split.

    The digest includes Hugging Face's dataset fingerprint, schema, row count,
    and dataset metadata.  It is intended to detect a changed dataset/cache
    during resume and experiment comparison; it is not a replacement for a
    separately archived raw-file checksum manifest.
    """
    if split is not None:
        dataset = get_dataset_split(dataset, split)

    if not isinstance(dataset, (Dataset, IterableDataset, DatasetDict)):
        raise TypeError(
            "Expected datasets.Dataset, datasets.IterableDataset, or "
            "datasets.DatasetDict, "
            f"got {type(dataset).__name__}"
        )

    if isinstance(dataset, DatasetDict):
        payload: object = {
            "splits": {
                split_name: _dataset_fingerprint_payload(split_dataset)
                for split_name, split_dataset in sorted(dataset.items())
            }
        }
    else:
        payload = _dataset_fingerprint_payload(dataset)

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def unified_dataset_loader(datasets: Sequence[DatasetSource | DatasetDict]) -> DatasetDict:
    """Combine dataset sources into one dataset.


    Args:
        datasets: Non-empty sequence of compatible Hugging Face datasets.

    Returns:
        A dataset containing the rows from each input dataset in order.

    Raises:
        ValueError: If no dataset sources are provided.
    """
    if not datasets:
        raise ValueError("unified_dataset_loader requires at least one dataset")

    splits: dict[str, list[Dataset]] = {
        split_name: [] for split_name in allowed_splits
    }

    dataset_names = {
        dataset_name
        for dataset in datasets
        for dataset_name in _dataset_names(dataset)
    }
    name = "+".join(sorted(dataset_names)) or None
    dataset_sizes = [get_dataset_size_bytes(dataset) for dataset in datasets]
    size_in_bytes = (
        sum(dataset_sizes)
        if all(dataset_size is not None for dataset_size in dataset_sizes)
        else None
    )

    for dataset in datasets:
        if not isinstance(dataset, (Dataset, IterableDataset, DatasetDict)):
            raise TypeError(
                "Expected datasets.Dataset, datasets.IterableDataset, or "
                "datasets.DatasetDict, "
                f"got {type(dataset).__name__}"
            )

        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                if split_name not in allowed_splits:
                    raise ValueError(
                        f"DatasetDict has an invalid split '{split_name}'. "
                        f"Allowed splits: {allowed_splits}"
                    )
                splits[split_name].append(split_dataset)
        else:
            splits["train"].append(dataset)

    info = (
        DatasetInfo(dataset_name=name, size_in_bytes=size_in_bytes)
        if name is not None or size_in_bytes is not None
        else None
    )

    return DatasetDict({
        split_name: concatenate_datasets(split_datasets, info=info)
        for split_name, split_datasets in splits.items()
        if split_datasets
    })


def get_dataset_split(
    dataset: Dataset | DatasetDict,
    split: str
) -> Dataset:
    """
    Resolve Dataset | DatasetDict into the Dataset used for training.

    Dataset:
        Used directly.

    DatasetDict:
        Uses the split.
    """

    if not split in allowed_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Allowed splits: {allowed_splits}"
        )

    if isinstance(dataset, Dataset):
        return dataset

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(
                f"DatasetDict has no '{split}' split. "
                f"Available splits: {list(dataset.keys())}"
            )

        return dataset[split]

    raise TypeError(
        "Expected datasets.Dataset or datasets.DatasetDict, "
        f"got {type(dataset).__name__}"
    )


def get_all_dataset(
        dataset: Dataset | DatasetDict,
):
    if isinstance(dataset, Dataset):
        return dataset

    if isinstance(dataset, DatasetDict):
        name = get_dataset_name(dataset)
        size_in_bytes = get_dataset_size_bytes(dataset)
        info = (
            DatasetInfo(dataset_name=name, size_in_bytes=size_in_bytes)
            if name is not None or size_in_bytes is not None
            else None
        )
        return concatenate_datasets(list(dataset.values()), info=info)

    raise TypeError(
        "Expected datasets.Dataset or datasets.DatasetDict, "
        f"got {type(dataset).__name__}"
    )

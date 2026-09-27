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

# This column is added while sources are combined so the training loop can
# recover the source boundaries without loading or tokenizing a source twice.
# The deliberately private-looking name avoids colliding with user text data.
DATASET_NAME_COLUMN = "__fraise_dataset_name__"

_SOURCE_NAME_SCAN_BATCH_SIZE = 65_536
_MAX_SOURCE_NAME_RANGES = 1_024


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


def _nonnegative_size(value: object) -> int | None:
    """Return a valid byte count, or ``None`` when metadata is unavailable."""
    return value if isinstance(value, int) and value >= 0 else None


def _dataset_source_size_bytes(dataset: DatasetSource) -> int | None:
    """Return the source/download size for one map-style dataset."""
    info = getattr(dataset, "info", None)
    download_size = _nonnegative_size(getattr(info, "download_size", None))
    if download_size is not None:
        return download_size

    # Datasets constructed or transformed locally may not retain the source
    # download metadata.  Their Arrow table size is the most useful fallback.
    if isinstance(dataset, Dataset):
        return int(dataset.data.nbytes)

    dataset_size = _nonnegative_size(getattr(info, "dataset_size", None))
    if dataset_size is not None:
        return dataset_size

    return _nonnegative_size(getattr(info, "size_in_bytes", None))


def _materialized_dataset_size_bytes(dataset: DatasetSource) -> int | None:
    """Return the size of the materialized Arrow table for one dataset."""
    if isinstance(dataset, Dataset):
        return int(dataset.data.nbytes)

    info = getattr(dataset, "info", None)
    return _nonnegative_size(getattr(info, "dataset_size", None))


def _sum_known_sizes(sizes: list[int | None]) -> int | None:
    """Sum byte counts only when every contributing source has one."""
    return sum(sizes) if all(size is not None for size in sizes) else None


def get_dataset_size_bytes(dataset: DatasetSource | DatasetDict) -> int | None:
    """Return the source/download size in bytes when it is available.

    Hugging Face's ``DatasetInfo.size_in_bytes`` is the sum of the download
    and materialized Arrow sizes.  It therefore reports roughly twice the
    source data for locally loaded datasets.  Prefer ``download_size`` here
    so provenance and training manifests describe the input data itself.
    """
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
        size = _dataset_source_size_bytes(source_dataset)
        if size is None:
            return None

        sizes.append(size)
        info = getattr(source_dataset, "info", None)
        sizes_from_info.append(
            _nonnegative_size(getattr(info, "download_size", None)) is not None
        )

    if not sizes:
        return None

    # DatasetDict splits loaded by Hugging Face commonly share one
    # DatasetInfo whose download_size already covers every split.  Only
    # de-duplicate when the info explicitly describes multiple splits; two
    # independent files can legitimately have the same byte size.
    shared_split_info = any(
        len(getattr(getattr(source_dataset, "info", None), "splits", {}) or {}) > 1
        for source_dataset in source_datasets
    )
    if shared_split_info and all(sizes_from_info) and len(set(sizes)) == 1:
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
        A dataset containing the rows from each input dataset in order. Named
        sources also carry :data:`DATASET_NAME_COLUMN` so token-based training
        can recover the source boundaries after concatenation.

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
    split_download_sizes: dict[str, list[int | None]] = {
        split_name: [] for split_name in allowed_splits
    }
    split_dataset_sizes: dict[str, list[int | None]] = {
        split_name: [] for split_name in allowed_splits
    }

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
                source_name = get_dataset_name(split_dataset)
                if source_name is None:
                    source_names = _dataset_names(dataset)
                    if len(source_names) == 1:
                        source_name = next(iter(source_names))
                splits[split_name].append(
                    _add_dataset_name_column(split_dataset, source_name)
                )
                split_download_sizes[split_name].append(
                    _dataset_source_size_bytes(split_dataset)
                )
                split_dataset_sizes[split_name].append(
                    _materialized_dataset_size_bytes(split_dataset)
                )
        else:
            source_name = get_dataset_name(dataset)
            splits["train"].append(
                _add_dataset_name_column(dataset, source_name)
            )
            split_download_sizes["train"].append(
                _dataset_source_size_bytes(dataset)
            )
            split_dataset_sizes["train"].append(
                _materialized_dataset_size_bytes(dataset)
            )

    combined_splits: dict[str, Dataset] = {}
    for split_name, split_datasets in splits.items():
        if not split_datasets:
            continue

        download_size = _sum_known_sizes(split_download_sizes[split_name])
        dataset_size = _sum_known_sizes(split_dataset_sizes[split_name])
        size_in_bytes = (
            download_size + dataset_size
            if download_size is not None and dataset_size is not None
            else None
        )
        info = (
            DatasetInfo(
                dataset_name=name,
                download_size=download_size,
                dataset_size=dataset_size,
                size_in_bytes=size_in_bytes,
            )
            if name is not None
            or download_size is not None
            or dataset_size is not None
            else None
        )
        combined_splits[split_name] = concatenate_datasets(
            split_datasets,
            info=info,
        )

    return DatasetDict(combined_splits)


def _add_dataset_name_column(
    dataset: Dataset,
    dataset_name: str | None,
) -> Dataset:
    """Keep the source name when map-style datasets are concatenated."""
    if dataset_name is None:
        return dataset

    if DATASET_NAME_COLUMN in dataset.column_names:
        existing_names = set(dataset[DATASET_NAME_COLUMN])
        if existing_names != {dataset_name}:
            raise ValueError(
                f"{DATASET_NAME_COLUMN!r} already contains source names "
                f"{sorted(existing_names)!r}; expected {dataset_name!r}"
            )
        return dataset

    return dataset.add_column(
        DATASET_NAME_COLUMN,
        [dataset_name] * len(dataset),
    )


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


def get_dataset_sources(
    dataset: Dataset | DatasetDict,
    split: str = "train",
) -> dict[str, Dataset]:
    """Return one map-style split per source dataset.

    ``unified_dataset_loader`` stores source names in
    :data:`DATASET_NAME_COLUMN`.  A standalone, named ``Dataset`` remains
    supported as a one-source input for callers and tests that do not use the
    unified loader.
    """
    source_split = get_dataset_split(dataset, split)

    if DATASET_NAME_COLUMN not in source_split.column_names:
        dataset_name = get_dataset_name(source_split)
        if dataset_name is None:
            raise ValueError(
                "training data must provide a non-empty dataset name; "
                f"add {DATASET_NAME_COLUMN!r} when combining unnamed sources"
            )
        return {dataset_name: source_split}

    source_ranges: dict[str, list[tuple[int, int]]] = {}
    source_names: dict[str, None] = {}
    collecting_ranges = True
    range_count = 0
    current_name: str | None = None
    range_start = 0
    row_index = 0

    # The unified loader concatenates sources, so its name column usually has
    # one contiguous range per source. Scan in bounded batches and represent
    # those ranges directly instead of materializing the full column and a
    # Python integer for every row.
    for batch in source_split.select_columns(DATASET_NAME_COLUMN).iter(
        batch_size=_SOURCE_NAME_SCAN_BATCH_SIZE
    ):
        for dataset_name in batch[DATASET_NAME_COLUMN]:
            if not isinstance(dataset_name, str) or not dataset_name:
                raise ValueError(
                    f"{DATASET_NAME_COLUMN!r} must contain non-empty strings; "
                    f"got {dataset_name!r} at row {row_index}"
                )
            source_names.setdefault(dataset_name, None)

            if dataset_name != current_name:
                if current_name is not None and collecting_ranges:
                    source_ranges.setdefault(current_name, []).append(
                        (range_start, row_index)
                    )
                    range_count += 1
                    if range_count > _MAX_SOURCE_NAME_RANGES:
                        # Shuffled or interleaved labels can have many runs.
                        # Drop the bounded range metadata and use Arrow-backed
                        # filters below rather than growing Python indexes.
                        source_ranges.clear()
                        collecting_ranges = False
                current_name = dataset_name
                range_start = row_index
            row_index += 1

    if current_name is not None and collecting_ranges:
        source_ranges.setdefault(current_name, []).append((range_start, row_index))
        range_count += 1
        if range_count > _MAX_SOURCE_NAME_RANGES:
            source_ranges.clear()
            collecting_ranges = False

    if collecting_ranges:
        source_splits: dict[str, Dataset] = {}
        for dataset_name, ranges in source_ranges.items():
            segments = [
                source_split.select(range(start, end))
                for start, end in ranges
            ]
            source_splits[dataset_name] = (
                segments[0]
                if len(segments) == 1
                else concatenate_datasets(segments)
            )
        return source_splits

    def matches_source(
        names: list[str],
        *,
        source_name: str,
    ) -> list[bool]:
        return [name == source_name for name in names]

    return {
        dataset_name: source_split.filter(
            matches_source,
            input_columns=DATASET_NAME_COLUMN,
            batched=True,
            batch_size=_SOURCE_NAME_SCAN_BATCH_SIZE,
            fn_kwargs={"source_name": dataset_name},
        )
        for dataset_name in source_names
    }


def get_all_dataset(
        dataset: Dataset | DatasetDict,
):
    if isinstance(dataset, Dataset):
        return dataset

    if isinstance(dataset, DatasetDict):
        name = get_dataset_name(dataset)
        download_size = get_dataset_size_bytes(dataset)
        dataset_size = _sum_known_sizes(
            [
                _materialized_dataset_size_bytes(split_dataset)
                for split_dataset in dataset.values()
            ]
        )
        size_in_bytes = (
            download_size + dataset_size
            if download_size is not None and dataset_size is not None
            else None
        )
        info = (
            DatasetInfo(
                dataset_name=name,
                download_size=download_size,
                dataset_size=dataset_size,
                size_in_bytes=size_in_bytes,
            )
            if name is not None or size_in_bytes is not None
            else None
        )
        return concatenate_datasets(list(dataset.values()), info=info)

    raise TypeError(
        "Expected datasets.Dataset or datasets.DatasetDict, "
        f"got {type(dataset).__name__}"
    )

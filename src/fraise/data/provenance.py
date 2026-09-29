"""Compact provenance for materialized datasets."""

import hashlib
import json

from datasets import Dataset


def describe_dataset(dataset: Dataset) -> dict:
    return {
        "rows": len(dataset),
        "size_in_bytes": dataset.info.download_size
        if dataset.info.download_size is not None
        else int(dataset.data.nbytes),
        "fingerprint": dataset._fingerprint,
    }


def dataset_fingerprint(sources: dict[str, Dataset]) -> str:
    payload = {name: describe_dataset(source) for name, source in sources.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

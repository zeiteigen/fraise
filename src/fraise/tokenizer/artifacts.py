"""Persist tokenizer artifacts and a single metadata format."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from fraise.data.provenance import dataset_fingerprint
from fraise.io import atomic_write, read_json, write_json

from .loading import load_tokenizer, tokenizer_fingerprint
from .training import MIN_FREQUENCY, train_tokenizer

SCHEMA_VERSION = 3


def read_metadata(directory: Path) -> list[dict]:
    path = directory / "metadata.json"
    if not path.exists():
        return []
    records = read_json(path)
    if not isinstance(records, list) or any(
        not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION
        for record in records
    ):
        raise ValueError(f"unsupported tokenizer metadata in {path}")
    return records


def create_tokenizer(
    sources, vocab_size: int, directory: Path, *, max_bytes_per_source=None
):
    request = {
        "dataset_fingerprint": dataset_fingerprint(sources),
        "sources": list(sources),
        "vocab_size": vocab_size,
        "min_frequency": MIN_FREQUENCY,
        "max_bytes_per_source": max_bytes_per_source,
    }
    return _create_artifact(
        directory,
        request,
        f"fraise-{vocab_size}",
        lambda: train_tokenizer(
            sources, vocab_size, max_bytes_per_source=max_bytes_per_source
        ),
    )


def create_macro_tokenizer(
    base,
    sources,
    vocab_size: int,
    directory: Path,
    *,
    max_macro_length: int,
    max_bytes_per_source=None,
    segmentation: str = "bpe",
    word_bounded_vocab_size: int | None = None,
    word_bounded: bool = False,
    whole_word_phrases: bool = False,
):
    """Train or reuse macro merges over ``base``; returns the artifact path."""
    from .macro_training import train_macro_tokenizer

    request = {
        "kind": "macro",
        "base_tokenizer_sha256": tokenizer_fingerprint(base),
        "dataset_fingerprint": dataset_fingerprint(sources),
        "sources": list(sources),
        "vocab_size": vocab_size,
        "max_macro_length": max_macro_length,
        "min_frequency": MIN_FREQUENCY,
        "max_bytes_per_source": max_bytes_per_source,
    }
    # Only non-default segmentations enter the request, so existing artifacts
    # keep matching their original requests.
    prefix = f"fraise-macro-{vocab_size}"
    if segmentation != "bpe":
        request["segmentation"] = segmentation
        request["word_bounded"] = word_bounded
        prefix += f"-{segmentation}" + ("-words" if word_bounded else "")
    if word_bounded_vocab_size is not None:
        request["word_bounded_vocab_size"] = word_bounded_vocab_size
        prefix += f"-words{word_bounded_vocab_size}"
    if whole_word_phrases:
        request["whole_word_phrases"] = True
        prefix += "-phrases"
    return _create_artifact(
        directory,
        request,
        prefix,
        lambda: train_macro_tokenizer(
            base,
            sources,
            vocab_size,
            max_macro_length=max_macro_length,
            max_bytes_per_source=max_bytes_per_source,
            segmentation=segmentation,
            word_bounded_vocab_size=word_bounded_vocab_size,
            word_bounded=word_bounded,
            whole_word_phrases=whole_word_phrases,
        ),
    )


def _create_artifact(directory: Path, request: dict, prefix: str, train):
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".metadata.lock")):
        records = read_metadata(directory)
        matches = [record for record in records if record.get("request") == request]
        if matches:
            record = matches[0]
            path = directory / record["path"]
            if path.is_file():
                tokenizer = load_tokenizer(path)
                if tokenizer_fingerprint(tokenizer) == record["tokenizer_sha256"]:
                    return path
        tokenizer = train()
        path = directory / f"{prefix}-{key}.json"
        atomic_write(path, lambda handle: handle.write(tokenizer.to_str()))
        record = {
            "schema_version": SCHEMA_VERSION,
            "path": path.name,
            "request": request,
            "actual_vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        records = [old for old in records if old.get("request") != request]
        records.append(record)
        write_json(directory / "metadata.json", records)
    return path

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
        raise ValueError(
            "legacy tokenizer metadata: run create-tokenizer --migrate-metadata"
        )
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
        tokenizer = train_tokenizer(
            sources, vocab_size, max_bytes_per_source=max_bytes_per_source
        )
        path = directory / f"fraise-{vocab_size}-{key}.json"
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

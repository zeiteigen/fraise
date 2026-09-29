"""Explicitly convert legacy tokenizer metadata while preserving artifact bytes."""

from pathlib import Path

from filelock import FileLock

from fraise.io import atomic_write, read_json, write_json

from .artifacts import SCHEMA_VERSION
from .loading import load_tokenizer, tokenizer_fingerprint


def migrate_metadata(directory: Path) -> int:
    """Convert the old index once; existing tokenizer bytes stay untouched."""
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".metadata.lock")):
        path = directory / "metadata.json"
        records = read_json(path) if path.exists() else []
        if isinstance(records, dict):
            records = records["tokenizers"]
        if isinstance(records, list) and all(
            record.get("schema_version") == SCHEMA_VERSION for record in records
        ):
            return len(records)
        migrated = []
        for record in records:
            if record.get("schema_version") == SCHEMA_VERSION:
                migrated.append(record)
                continue
            old_path = record.get("path")
            if not old_path:
                raise ValueError(
                    "legacy record has no path; specify its tokenizer file explicitly"
                )
            artifact = Path(old_path)
            if not artifact.is_absolute():
                artifact = directory / artifact
            tokenizer = load_tokenizer(artifact)
            migrated.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "path": str(artifact.resolve()),
                    "request": None,
                    "actual_vocab_size": tokenizer.get_vocab_size(),
                    "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                    "legacy": record,
                }
            )
        if path.exists():
            atomic_write(
                directory / "metadata.v2.json",
                lambda handle: handle.write(path.read_text()),
            )
        write_json(path, migrated)
        return len(migrated)

"""Atomic writes and JSON reads for local artifacts."""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, write: Callable[[object], None], *, binary=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(
            descriptor, "wb" if binary else "w", encoding=None if binary else "utf-8"
        ) as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def write_json(path: Path, value: object) -> Path:
    return atomic_write(
        path,
        lambda handle: handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n"),
    )


def read_json(path: str | Path):
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        return json.load(handle)

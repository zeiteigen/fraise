"""Resolve the shared data root for commands and notebooks."""

import os
from pathlib import Path


def data_root(path: str | Path | None = None) -> Path:
    return (
        Path(path or os.environ.get("FRAISE_DATA_DIR", "data")).expanduser().resolve()
    )

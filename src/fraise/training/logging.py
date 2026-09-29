"""Append run messages without rewriting run metadata."""

from datetime import datetime, timezone
from pathlib import Path


class RunLog:
    def __init__(self, directory: Path):
        self.path = directory / "training.log"

    def __call__(self, message: str):
        print(message)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

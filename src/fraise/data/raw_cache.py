"""Request-specific raw text artifacts."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from fraise.io import read_json, write_json


@dataclass(frozen=True)
class RawCache:
    root: Path
    request: dict

    @property
    def directory(self) -> Path:
        digest = hashlib.sha256(
            json.dumps(self.request, sort_keys=True).encode()
        ).hexdigest()[:24]
        return self.root / digest

    @property
    def path(self) -> Path:
        return self.directory / "text.jsonl"

    def ready(self) -> bool:
        manifest = self.directory / "request.json"
        if not self.path.is_file() or not manifest.is_file():
            return False
        try:
            return read_json(manifest) == self.request
        except OSError, ValueError:
            return False

    def complete(self) -> None:
        write_json(self.directory / "request.json", self.request)

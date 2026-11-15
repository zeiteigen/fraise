"""Autocomplete documents kept between sessions (fraise-web kept them in browser storage)."""

import json
from pathlib import Path
from time import time
from uuid import uuid4

from fraise.io import read_json, write_json

LIMIT = 40
MAX_BYTES = 2_000_000


class SavedInputs:
    """The newest ``LIMIT`` inputs, newest first, in one JSON file."""

    def __init__(self, path):
        self.path = Path(path)

    def items(self):
        try:
            items = read_json(self.path)
        except (OSError, ValueError):
            return []
        if not isinstance(items, list):
            return []
        return [
            item
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("text"), str)
            and isinstance(item.get("ref"), dict)
        ][:LIMIT]

    def remember(self, item):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("a saved input needs its text")
        if not isinstance(item.get("ref"), dict) or not item["ref"].get("run_id"):
            raise ValueError("a saved input needs its checkpoint")
        item = {**item, "id": item.get("id") or uuid4().hex, "updated": time()}
        if len(json.dumps(item, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
            raise ValueError("this input is too large to save")
        items = [item] + [previous for previous in self.items() if previous["id"] != item["id"]]
        self._write(items[:LIMIT])
        return item

    def clear(self):
        self._write([])

    def _write(self, items):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, items)

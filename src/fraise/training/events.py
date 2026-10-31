"""Append structured training events for local job observers."""

import json
from pathlib import Path
from threading import Lock
from time import time


class EventWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def __call__(self, event):
        payload = json.dumps({"time": time(), **event}, allow_nan=False)
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


def emit(callback, kind, **values):
    if callback is not None:
        callback({"type": kind, **values})

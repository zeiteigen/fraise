"""Durable job descriptors and bounded incremental progress reads."""

import json
from collections import deque
from pathlib import Path

from fraise.io import read_json, write_json

from .catalog import artifact_name, contained

TERMINAL = {"completed", "stopped", "failed", "interrupted"}


class JobStorage:
    def __init__(self, root):
        self.root = Path(root)
        self._events = {}

    def directory(self, identifier):
        return contained(self.root / artifact_name(identifier), self.root)

    def save(self, job):
        write_json(self.directory(job["id"]) / "job.json", job)

    def get(self, identifier):
        path = self.directory(identifier) / "job.json"
        if not path.is_file():
            raise FileNotFoundError("training job was not found")
        return read_json(path)

    def jobs(self):
        records = []
        for path in sorted(self.root.glob("*/job.json"), reverse=True):
            try:
                contained(path, self.root)
                records.append(read_json(path))
            except OSError, ValueError:
                continue
        return records

    def events(self, identifier):
        path = self.directory(identifier) / "events.jsonl"
        offset, history = self._events.get(identifier, (0, deque(maxlen=1000)))
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(offset)
                while line := handle.readline():
                    if not line.endswith(b"\n"):
                        break
                    offset = handle.tell()
                    try:
                        history.append(json.loads(line))
                    except ValueError, UnicodeError:
                        continue
        self._events[identifier] = (offset, history)
        return list(history)

    def snapshot(self, identifier, cursor=0):
        job = self.get(identifier)
        events = self.events(identifier)
        path = self.directory(identifier) / "output.log"
        logs = ""
        if path.exists():
            with path.open("rb") as handle:
                # Initial reads and lagging clients receive a bounded tail.
                size = path.stat().st_size
                handle.seek(max(min(cursor, size), size - 131072))
                logs = handle.read(131072).decode("utf-8", errors="replace")
                cursor = handle.tell()
        return {**job, "events": events, "logs": logs, "cursor": cursor}

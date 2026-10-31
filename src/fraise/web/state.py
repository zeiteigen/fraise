"""Serialize access to the compute device across web operations."""

from threading import Lock


class BusyError(RuntimeError):
    pass


class ComputeGate:
    def __init__(self):
        self._lock = Lock()
        self._busy = None

    @property
    def busy(self):
        with self._lock:
            return dict(self._busy) if self._busy else None

    def acquire(self, kind, identifier):
        with self._lock:
            if self._busy:
                raise BusyError(f"{self._busy['kind'].capitalize()} is already running")
            self._busy = {"kind": kind, "id": identifier}

    def release(self, identifier):
        with self._lock:
            if self._busy and self._busy["id"] == identifier:
                self._busy = None

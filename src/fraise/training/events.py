"""Append structured training events for local job observers."""

import json
import math
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


def _numbers(values, prefix=""):
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _numbers(value, f"{name}/")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, value


class AblationKitMetrics:
    """Log events' numbers as metrics of the AblationKit job's MLflow run, at ``global_step``.

    Names match what the job's metrics-file collection makes of the same events (nested objects
    joined with ``/``; ``time`` and ``global_step`` are not metrics), so runs launched either way
    compare. AblationKit puts ``ablationkit_sdk`` on every job's path; outside a job it is
    missing or does nothing.
    """

    def __init__(self):
        try:
            import ablationkit_sdk
        except ImportError:
            ablationkit_sdk = None
        self.sdk = ablationkit_sdk
        self.lock = Lock()

    def __call__(self, event):
        step = event.get("global_step")
        if self.sdk is None or isinstance(step, bool) or not isinstance(step, int):
            return
        metrics = {
            name: value
            for name, value in _numbers(event)
            if name not in ("time", "global_step") and math.isfinite(value)
        }
        if metrics:
            with self.lock:
                self.sdk.log_metrics(metrics, step=step)


def fan_out(*callbacks):
    """One event callback that calls each of ``callbacks`` that is not None, or None."""
    callbacks = [callback for callback in callbacks if callback is not None]
    if not callbacks:
        return None

    def call(event):
        for callback in callbacks:
            callback(event)

    return call

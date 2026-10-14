"""Evaluate every checkpoint with bounded pending work and deadlines."""

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from .deadline import ValidationTimeout


@dataclass
class ValidationJob:
    step: int
    path: Path
    future: Future
    submitted_at: float


class ValidationQueue:
    """One active checkpoint and one waiting checkpoint, with backpressure.

    ``timeout_seconds`` is reported in logs; ``submit`` must return futures that
    enforce it.
    """

    def __init__(self, submit, log, *, timeout_seconds=300, reset=lambda: None):
        self._submit = submit
        self._log = log
        self._timeout = timeout_seconds
        self._reset = reset
        self._active: ValidationJob | None = None
        self._queued: tuple[int, Path] | None = None
        self.latest: tuple[int, float] | None = None
        self.failures: list[dict] = []
        self.results: list[dict] = []
        self.wait_seconds = 0.0

    def _failure(self, step, error, status, elapsed):
        self.failures.append({"step": step, "error": error})
        self.results.append(
            {"step": step, "status": status, "error": error, "elapsed_seconds": elapsed}
        )
        self._log(f"Validation {status} at global step {step}: {error}")

    def _start(self, step, path):
        started = monotonic()
        try:
            future = self._submit(step, path)
        except Exception as exc:
            self._failure(step, str(exc), "submission failed", monotonic() - started)
            self._reset()
        else:
            self._active = ValidationJob(step, path, future, started)
            self._log(
                f"Validation submitted at global step {step}: {path.name}; "
                f"deadline {self._timeout:g}s."
            )

    def queue(self, step, path):
        self.report()
        if self._queued is not None:
            self._log(
                f"Validation queue full; training waiting before queuing global step {step}."
            )
            started = monotonic()
            try:
                self._report_one(wait=True)
            finally:
                waited = monotonic() - started
                self.wait_seconds += waited
                self._log(
                    f"Validation queue wait: {waited:.3f}s "
                    f"(total {self.wait_seconds:.3f}s)."
                )
        if self._active is None:
            self._start(step, path)
        else:
            self._queued = (step, path)
            self._log(
                f"Validation queued at global step {step}; waiting for step {self._active.step}."
            )

    def _report_one(self, *, wait):
        # Submitted futures carry their own deadline (see deadline.timed_future),
        # which kills the worker independently of this thread, so waiting here
        # is always bounded.
        job = self._active
        if not wait and not job.future.done():
            return False
        try:
            loss = job.future.result()
        except ValidationTimeout as exc:
            self._failure(
                job.step, str(exc), "timed out", monotonic() - job.submitted_at
            )
            self._reset()
        except Exception as exc:
            self._failure(job.step, str(exc), "failed", monotonic() - job.submitted_at)
            self._reset()
        else:
            elapsed = monotonic() - job.submitted_at
            self.results.append(
                {
                    "step": job.step,
                    "status": "completed" if loss is not None else "unavailable",
                    "loss": loss,
                    "elapsed_seconds": elapsed,
                }
            )
            if loss is None:
                self._log(
                    f"Validation loss unavailable at global step {job.step} ({elapsed:.1f}s)."
                )
            else:
                self.latest = (job.step, loss)
                self._log(
                    f"Validation loss at global step {job.step}: {loss:.6f} ({elapsed:.1f}s)."
                )
        self._active = None
        if self._queued is not None:
            step, path = self._queued
            self._queued = None
            self._start(step, path)
        return True

    def report(self, *, wait=False):
        while self._active is not None and self._report_one(wait=wait):
            pass

    def cancel(self):
        if self._active is not None:
            self._failure(
                self._active.step,
                "training exited before validation completed",
                "cancelled",
                monotonic() - self._active.submitted_at,
            )
            self._active = None
        if self._queued is not None:
            self._failure(
                self._queued[0],
                "training exited before validation started",
                "cancelled",
                0,
            )
            self._queued = None

    def protected_paths(self):
        paths = {self._active.path} if self._active is not None else set()
        if self._queued is not None:
            paths.add(self._queued[1])
        return paths

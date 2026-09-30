"""Coordinate checkpoint validation, durable outcomes, and process cleanup."""

from .validation_process import ValidationProcess
from .validation_queue import ValidationQueue


class ValidationRunner:
    def __init__(self, config, split, settings, pad_id, log, *, on_update=None):
        self.process = None
        self.queue = None
        self._on_update = on_update
        self._recorded = 0
        if settings.max_validation_tokens == 0:
            log("Validation disabled: max_validation_tokens is 0.")
        elif split is None:
            log("Validation disabled: no held-out split.")
        else:
            self.process = ValidationProcess(config, split, settings, pad_id, log)
            self.queue = ValidationQueue(
                self.process.submit,
                log,
                timeout_seconds=settings.validation_timeout_seconds,
                reset=self.process.stop,
            )
            log(
                f"Validation enabled: backend {settings.validation_device}, target budget {settings.max_validation_tokens}, "
                f"deadline {settings.validation_timeout_seconds:g}s per checkpoint; every checkpoint queued."
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if self.queue is not None:
                self.queue.cancel()
                self._persist()
        finally:
            if self.process is not None:
                self.process.stop(graceful=exc[0] is None)

    def _persist(self):
        if self.queue is not None and len(self.queue.results) != self._recorded:
            if self._on_update is not None:
                self._on_update(
                    validation=self.latest,
                    validation_failures=self.failures,
                    validation_results=list(self.queue.results),
                )
            self._recorded = len(self.queue.results)

    def submit(self, step, path):
        if self.queue is not None:
            self.queue.queue(step, path)
            self._persist()

    def report(self, *, wait=False):
        if self.queue is not None:
            self.queue.report(wait=wait)
            self._persist()

    @property
    def protected_paths(self):
        return self.queue.protected_paths() if self.queue is not None else set()

    @property
    def latest(self):
        return self.queue.latest if self.queue is not None else None

    @property
    def failures(self):
        return self.queue.failures if self.queue is not None else []

"""Own the bounded validation queue and its process lifetime."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from .validation_queue import ValidationQueue
from .validation_worker import evaluate_checkpoint, init_validation_worker


class ValidationRunner:
    def __init__(self, config, split, settings, pad_id, log):
        self.executor = None
        self.queue = None
        if split is not None:
            self.executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=init_validation_worker,
                initargs=(config, split, settings, pad_id),
            )
            self.queue = ValidationQueue(
                lambda path: self.executor.submit(evaluate_checkpoint, str(path)), log
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.executor is not None:
            self.executor.shutdown(cancel_futures=True)

    def submit(self, step, path):
        if self.queue is not None:
            self.queue.queue(step, path)

    def report(self, *, wait=False):
        if self.queue is not None:
            self.queue.report(wait=wait)

    @property
    def protected_paths(self):
        return self.queue.protected_paths() if self.queue is not None else set()

    @property
    def latest(self):
        return self.queue.latest if self.queue is not None else None

    @property
    def failures(self):
        return self.queue.failures if self.queue is not None else []

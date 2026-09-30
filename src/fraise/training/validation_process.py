"""Own a spawned evaluator and relay its progress independently of training."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from threading import Event, Thread

from .validation_deadline import timed_future
from .validation_worker import evaluate_checkpoint, init_validation_worker


class ValidationProcess:
    def __init__(self, config, split, settings, pad_id, log):
        self._arguments = (config, split, settings, pad_id, os.getpid())
        self._timeout = settings.validation_timeout_seconds
        self._log = log
        self.executor = None
        self._listener = None

    def _start(self):
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        self._receiver = receiver
        self._sender = sender
        self._stop = Event()
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=init_validation_worker,
            initargs=(*self._arguments, sender),
        )
        self._listener = Thread(
            target=self._relay,
            args=(receiver, self._stop),
            daemon=True,
            name="validation-progress",
        )
        self._listener.start()

    def _relay(self, receiver, stop):
        while not stop.is_set() or receiver.poll():
            try:
                if receiver.poll(0.1):
                    self._log(receiver.recv())
            except EOFError, OSError:
                return

    def submit(self, step, path):
        if self.executor is None:
            self._start()
        executor = self.executor
        future = executor.submit(evaluate_checkpoint, step, str(path))

        def kill_worker():
            self._log(
                f"Validation step {step}: deadline reached; killing worker independently of training."
            )
            executor.kill_workers()

        return timed_future(future, self._timeout, kill_worker)

    def stop(self, *, graceful=False):
        if self.executor is None:
            return
        executor = self.executor
        if graceful:
            shutdown = Thread(
                target=executor.shutdown, kwargs={"cancel_futures": True}, daemon=True
            )
            shutdown.start()
            shutdown.join(timeout=2)
            if shutdown.is_alive():
                executor.kill_workers()
                shutdown.join(timeout=1)
        else:
            # Python 3.14's public API avoids an unbounded wait on stuck native code.
            executor.kill_workers()
        self.executor = None
        self._stop.set()
        self._listener.join(timeout=1)
        self._receiver.close()
        self._sender.close()

"""Enforce a worker deadline even while the training thread is busy."""

from concurrent.futures import Future, TimeoutError
from threading import Lock, Timer


class ValidationTimeout(TimeoutError):
    pass


def timed_future(source, timeout_seconds, kill_worker):
    result = Future()
    lock = Lock()

    def expire():
        with lock:
            if result.done():
                return
            result.set_exception(
                ValidationTimeout(
                    f"exceeded {timeout_seconds:g}s deadline; worker killed"
                )
            )
        kill_worker()

    timer = Timer(timeout_seconds, expire)
    timer.daemon = True

    def complete(future):
        timer.cancel()
        with lock:
            if result.done():
                return
            try:
                value = future.result()
            except Exception as exc:
                result.set_exception(exc)
            else:
                result.set_result(value)

    source.add_done_callback(complete)
    timer.start()
    return result

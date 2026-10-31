"""Translate local Stop requests into safe training-boundary cancellation."""

import signal
from contextlib import contextmanager


@contextmanager
def stop_signals(stop_file, callback):
    """Interrupt preparation, but never interrupt a running optimizer update."""
    training = False

    def on_event(event):
        nonlocal training
        if event.get("type") == "stage" and event.get("phase") == "training":
            training = True
        if callback is not None:
            callback(event)

    def interrupt(signum, frame):
        stop_file.touch()
        if not training:
            raise KeyboardInterrupt

    previous = None
    if stop_file is not None:
        previous = signal.signal(signal.SIGINT, interrupt)
    try:
        yield on_event
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

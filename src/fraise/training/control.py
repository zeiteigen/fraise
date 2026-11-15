"""Translate stop requests into safe training-boundary cancellation."""

import signal
from contextlib import contextmanager


@contextmanager
def stop_signals(stop_file, callback):
    """Save and stop after the current optimizer update when asked; never interrupt one.

    SIGTERM (an AblationKit job's Cancel or time limit) always asks; SIGINT asks when a stop
    file is given (fraise-web's Stop), and otherwise keeps its usual meaning. Before training
    starts, a request interrupts preparation instead. Yields ``(on_event, stop_requested)``.
    """
    training = False
    requested = False

    def on_event(event):
        nonlocal training
        if event.get("type") == "stage" and event.get("phase") == "training":
            training = True
        if callback is not None:
            callback(event)

    def interrupt(signum, frame):
        nonlocal requested
        requested = True
        if stop_file is not None:
            stop_file.touch()
        if not training:
            raise KeyboardInterrupt

    def stop_requested():
        return requested or (stop_file is not None and stop_file.exists())

    handled = [signal.SIGTERM] + ([signal.SIGINT] if stop_file is not None else [])
    previous = {number: signal.signal(number, interrupt) for number in handled}
    try:
        yield on_event, stop_requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)

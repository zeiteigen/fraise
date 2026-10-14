"""Prevent a Linux validation worker from surviving its training parent."""

import ctypes
import os
import signal
import sys


def exit_with_parent(parent_pid):
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Cover the race where the parent exited before the signal was installed.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)

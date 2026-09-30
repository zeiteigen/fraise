import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fixtures import tiny_config, tiny_settings

from fraise.training.validation_deadline import ValidationTimeout
from fraise.training.validation_process import ValidationProcess


class ValidationProcessTests(unittest.TestCase):
    def test_deadline_kills_real_worker_and_allows_restart(self):
        process = ValidationProcess(
            tiny_config(),
            [],
            tiny_settings(validation_timeout_seconds=0.1),
            0,
            lambda message: None,
        )
        try:
            for step in (1, 2):
                future = process.submit(step, Path("unused.pt"))
                workers = list(process.executor._processes.values())
                with self.assertRaises(ValidationTimeout):
                    future.result(timeout=5)
                process.stop()
                for worker in workers:
                    worker.join(timeout=3)
                    self.assertFalse(worker.is_alive())
        finally:
            process.stop()

    def test_context_cleanup_kills_worker_without_waiting_for_deadline(self):
        process = ValidationProcess(
            tiny_config(),
            [],
            tiny_settings(validation_timeout_seconds=60),
            0,
            lambda message: None,
        )
        try:
            process.submit(1, Path("unused.pt"))
            workers = list(process.executor._processes.values())
            started = time.monotonic()
            process.stop()
            for worker in workers:
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
            self.assertLess(time.monotonic() - started, 4)
        finally:
            process.stop()

    @unittest.skipUnless(sys.platform == "linux", "Linux parent-death signal")
    def test_worker_exits_when_parent_disappears(self):
        with TemporaryDirectory() as directory:
            script = Path(directory) / "parent.py"
            script.write_text("""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from fraise.training.worker_lifetime import exit_with_parent

def run(parent_pid):
    exit_with_parent(parent_pid)
    return os.getpid()

if __name__ == "__main__":
    executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
    pid = executor.submit(run, os.getpid()).result(timeout=10)
    print(pid, flush=True)
    time.sleep(60)
""")
            parent = subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            worker_pid = None
            try:
                import select

                self.assertTrue(
                    select.select([parent.stdout], [], [], 15)[0],
                    "worker never started",
                )
                worker_pid = int(parent.stdout.readline())
                parent.kill()
                parent.communicate(timeout=5)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    status = Path(f"/proc/{worker_pid}/stat")
                    if not status.exists() or status.read_text().split(") ", 1)[
                        1
                    ].startswith("Z "):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("worker survived the training parent's death")
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.communicate(timeout=5)
                if worker_pid is not None:
                    try:
                        os.kill(worker_pid, 9)
                    except ProcessLookupError:
                        pass

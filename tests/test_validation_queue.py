import unittest
from concurrent.futures import Future
from pathlib import Path

from fraise.training.validation_queue import ValidationQueue


class ValidationQueueTests(unittest.TestCase):
    def test_every_checkpoint_is_evaluated_with_bounded_backpressure(self):
        from threading import Timer

        submitted = []
        messages = []

        def submit(step, path):
            future = Future()
            if step == 3:
                future.set_result(0.5)
            submitted.append((path, future))
            return future

        queue = ValidationQueue(submit, messages.append)
        first, second, third = (Path(f"step-{step}.pt") for step in (1, 2, 3))
        queue.queue(1, first)
        queue.queue(2, second)
        self.assertEqual(queue.protected_paths(), {first, second})
        timer = Timer(0.02, submitted[0][1].set_result, args=(1.0,))
        timer.start()
        queue.queue(3, third)
        timer.join()
        self.assertEqual([path for path, _ in submitted], [first, second])
        self.assertEqual(queue.protected_paths(), {second, third})
        self.assertTrue(any("training waiting" in message for message in messages))
        submitted[1][1].set_result(0.75)
        queue.report(wait=True)
        self.assertEqual([path for path, _ in submitted], [first, second, third])
        self.assertEqual(queue.latest, (3, 0.5))
        self.assertEqual([record["step"] for record in queue.results], [1, 2, 3])
        self.assertEqual(queue.protected_paths(), set())

    def test_timeout_releases_checkpoint_and_starts_waiting_job(self):
        from unittest.mock import patch

        submitted = []
        resets = []

        def submit(step, path):
            future = Future()
            submitted.append(future)
            return future

        queue = ValidationQueue(
            submit,
            lambda message: None,
            timeout_seconds=5,
            reset=lambda: resets.append(True),
        )
        with patch("fraise.training.validation_queue.monotonic", return_value=0):
            queue.queue(1, Path("first.pt"))
            queue.queue(2, Path("second.pt"))
        with patch("fraise.training.validation_queue.monotonic", return_value=6):
            queue.report()
        self.assertEqual(resets, [True])
        self.assertEqual(queue.results[0]["status"], "timed out")
        self.assertEqual(queue.protected_paths(), {Path("second.pt")})
        submitted[1].set_result(0.5)
        queue.report(wait=True)
        self.assertEqual(queue.latest, (2, 0.5))

    def test_cancel_records_both_active_and_waiting_checkpoints(self):
        queue = ValidationQueue(lambda step, path: Future(), lambda message: None)
        queue.queue(1, Path("first.pt"))
        queue.queue(2, Path("second.pt"))
        queue.cancel()
        self.assertEqual(
            [record["status"] for record in queue.results], ["cancelled", "cancelled"]
        )
        self.assertEqual(queue.protected_paths(), set())


if __name__ == "__main__":
    unittest.main()


class ValidationFailureTests(unittest.TestCase):
    def test_backend_timeout_is_reported_with_its_original_error(self):
        future = Future()
        future.set_exception(TimeoutError("NPU inference request timed out"))
        queue = ValidationQueue(lambda step, path: future, lambda message: None)
        queue.queue(1, Path("first.pt"))
        queue.report()
        self.assertEqual(queue.results[0]["status"], "failed")
        self.assertEqual(queue.failures[0]["error"], "NPU inference request timed out")

    def test_worker_failure_is_recorded_and_the_next_checkpoint_still_runs(self):
        futures = []

        def submit(step, path):
            future = Future()
            futures.append(future)
            return future

        queue = ValidationQueue(submit, lambda message: None)
        queue.queue(1, Path("first.pt"))
        queue.queue(2, Path("second.pt"))
        futures[0].set_exception(RuntimeError("broken evaluator"))
        queue.report()
        self.assertEqual(queue.failures, [{"step": 1, "error": "broken evaluator"}])
        futures[1].set_result(0.5)
        queue.report(wait=True)
        self.assertEqual(queue.latest, (2, 0.5))

    def test_submission_failure_is_recorded(self):
        def submit(step, path):
            raise RuntimeError("worker unavailable")

        queue = ValidationQueue(submit, lambda message: None)
        queue.queue(1, Path("first.pt"))
        self.assertEqual(queue.failures, [{"step": 1, "error": "worker unavailable"}])

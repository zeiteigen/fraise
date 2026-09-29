import unittest
from concurrent.futures import Future
from pathlib import Path

from fraise.training.validation_queue import ValidationQueue


class ValidationQueueTests(unittest.TestCase):
    def test_only_latest_waiting_checkpoint_is_evaluated(self):
        submitted: list[tuple[Path, Future[float | None]]] = []
        messages: list[str] = []

        def submit(path: Path) -> Future[float | None]:
            future: Future[float | None] = Future()
            submitted.append((path, future))
            return future

        queue = ValidationQueue(submit, messages.append)
        first, skipped, latest = (Path(f"step-{step}.pt") for step in (1, 2, 3))
        queue.queue(1, first)
        queue.queue(2, skipped)
        queue.queue(3, latest)

        self.assertEqual([path for path, _ in submitted], [first])
        self.assertEqual(queue.protected_paths(), {first, latest})

        submitted[0][1].set_result(1.0)
        queue.report()
        self.assertEqual([path for path, _ in submitted], [first, latest])
        self.assertEqual(queue.latest, (1, 1.0))

        submitted[1][1].set_result(0.5)
        queue.report(wait=True)
        self.assertEqual(queue.latest, (3, 0.5))
        self.assertEqual(queue.protected_paths(), set())
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()


class ValidationFailureTests(unittest.TestCase):
    def test_worker_failure_is_recorded_and_the_next_checkpoint_still_runs(self):
        futures = []

        def submit(path):
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
        def submit(path):
            raise RuntimeError("worker unavailable")

        queue = ValidationQueue(submit, lambda message: None)
        queue.queue(1, Path("first.pt"))
        self.assertEqual(queue.failures, [{"step": 1, "error": "worker unavailable"}])

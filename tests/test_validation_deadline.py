import unittest
from concurrent.futures import Future
from threading import Event

from fraise.training.validation_deadline import ValidationTimeout, timed_future


class ValidationDeadlineTests(unittest.TestCase):
    def test_deadline_kills_worker_without_training_polling(self):
        killed = Event()
        source = Future()
        future = timed_future(source, 0.02, killed.set)
        self.assertTrue(killed.wait(2))
        with self.assertRaises(ValidationTimeout):
            future.result()
        # A late worker result must not overwrite the timeout or raise in its callback.
        source.set_result(0.5)
        with self.assertRaises(ValidationTimeout):
            future.result()

    def test_completed_result_cancels_deadline(self):
        killed = Event()
        source = Future()
        future = timed_future(source, 0.02, killed.set)
        source.set_result(0.5)
        self.assertEqual(future.result(), 0.5)
        self.assertFalse(killed.wait(0.05))

    def test_worker_exception_is_preserved(self):
        source = Future()
        future = timed_future(source, 1, lambda: None)
        source.set_exception(RuntimeError("compile failed"))
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            future.result()

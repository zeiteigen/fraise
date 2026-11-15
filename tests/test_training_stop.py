"""Graceful stopping preserves completed updates and interrupts validation waits."""

import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch

import torch
from fixtures import eager_loss, tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.io import read_json
from fraise.training.run.checkpoints import load_checkpoint_file
from fraise.training.runtime import TrainingRuntime
from fraise.training.training import pretrain_model
from fraise.training.validation.queue import ValidationQueue


class WaitingFuture(Future):
    """A real unresolved future that exposes when a bounded wait has begun."""

    def __init__(self):
        super().__init__()
        self.waiting = Event()

    def result(self, timeout=None):
        if timeout is not None:
            self.waiting.set()
        return super().result(timeout)


class TrainingStopTests(unittest.TestCase):
    def setUp(self):
        cache = TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        cache_patch = patch("datasets.config.HF_DATASETS_CACHE", Path(cache.name))
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        offline = patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"})
        offline.start()
        self.addCleanup(offline.stop)
        compile_patch = eager_loss()
        compile_patch.start()
        self.addCleanup(compile_patch.stop)

    def assert_payload_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key, value in expected.items():
                self.assert_payload_equal(actual[key], value)
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_payload_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def train(self, directory, name, settings, **kwargs):
        return pretrain_model(
            tiny_tokenizer(),
            tiny_data(),
            settings,
            tiny_config(),
            directory,
            name,
            device="cpu",
            **kwargs,
        )

    def test_stop_and_resume_match_uninterrupted_at_every_training_boundary(self):
        settings = tiny_settings(
            dataset_token_counts={"fixture": 6},
            max_train_tokens=6,
            deterministic=True,
            log_interval_steps=1,
            checkpoint_keep_last=4,
        )
        with TemporaryDirectory() as directory:
            expected = self.train(directory, "reference", settings)
            reference = Path(directory) / "reference"
            final_checkpoint = load_checkpoint_file(
                reference / "checkpoint_step_00000003.pt"
            )
            cases = (
                ("mid-run", "progress", 1),
                ("checkpoint", "checkpoint", 2),
                ("final-step", "checkpoint", 3),
            )
            for name, event_type, step in cases:
                with self.subTest(boundary=name):
                    stop = Event()
                    events = []

                    def on_event(
                        event,
                        *,
                        events=events,
                        event_type=event_type,
                        step=step,
                        stop=stop,
                    ):
                        events.append(event)
                        if (
                            event["type"] == event_type
                            and event.get("global_step") == step
                        ):
                            stop.set()

                    stopped = self.train(
                        directory,
                        name,
                        settings,
                        stop_requested=stop.is_set,
                        on_event=on_event,
                    )
                    run = Path(directory) / name
                    path = run / f"checkpoint_step_{step:08d}.pt"
                    checkpoint = load_checkpoint_file(path)
                    expected_checkpoint = load_checkpoint_file(reference / path.name)
                    for key in (
                        "model_state_dict",
                        "optimizer_state_dict",
                        "scheduler_state_dict",
                        "rng_state",
                        "training_state",
                    ):
                        self.assert_payload_equal(
                            checkpoint[key], expected_checkpoint[key]
                        )
                    self.assertEqual(checkpoint["training_state"]["global_step"], step)
                    self.assertEqual(checkpoint["training_state"]["tokens_seen"], 2 * step)
                    self.assertEqual(
                        checkpoint["scheduler_state_dict"]["last_epoch"], step
                    )
                    self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
                    self.assert_payload_equal(
                        stopped.state_dict(), checkpoint["model_state_dict"]
                    )
                    metadata = read_json(run / "model_meta.json")
                    self.assertEqual(metadata["training_status"], "stopped")
                    self.assertFalse(metadata["training_finished"])
                    self.assertFalse((run / f"{name}.pt").exists())
                    self.assertEqual(events[-1]["type"], "stopped")
                    self.assertEqual(events[-1]["global_step"], step)

                    resumed = self.train(directory, name, settings, resume_from=path)
                    self.assert_payload_equal(
                        resumed.state_dict(), expected.state_dict()
                    )
                    resumed_checkpoint = load_checkpoint_file(
                        run / "checkpoint_step_00000003.pt"
                    )
                    self.assert_payload_equal(resumed_checkpoint, final_checkpoint)
                    self.assertTrue(
                        read_json(run / "model_meta.json")["training_finished"]
                    )

    def test_stop_after_an_update_resumes_unfinished_bookkeeping(self):
        settings = tiny_settings(
            dataset_token_counts={"fixture": 6},
            max_train_tokens=6,
            deterministic=True,
            checkpoint_keep_last=4,
        )
        with TemporaryDirectory() as directory:
            self.train(directory, "reference", settings)
            reference = Path(directory) / "reference"
            expected_final = load_checkpoint_file(
                reference / "checkpoint_step_00000003.pt"
            )
            for target in (2, 3):
                with self.subTest(completed_update=target):
                    stop = Event()
                    completed = []
                    original_step = TrainingRuntime.step

                    def step_and_stop(
                        runtime,
                        batch,
                        *,
                        target=target,
                        stop=stop,
                        completed=completed,
                        original_step=original_step,
                    ):
                        result = original_step(runtime, batch)
                        completed.append(result)
                        if len(completed) == target:
                            stop.set()
                        return result

                    name = f"after-update-{target}"
                    with patch.object(TrainingRuntime, "step", step_and_stop):
                        self.train(
                            directory, name, settings, stop_requested=stop.is_set
                        )
                    run = Path(directory) / name
                    checkpoint_path = run / f"checkpoint_step_{target:08d}.pt"
                    checkpoint = load_checkpoint_file(checkpoint_path)
                    reference_checkpoint = load_checkpoint_file(
                        reference / checkpoint_path.name
                    )
                    for key in (
                        "model_state_dict",
                        "optimizer_state_dict",
                        "scheduler_state_dict",
                        "rng_state",
                    ):
                        self.assert_payload_equal(
                            checkpoint[key], reference_checkpoint[key]
                        )
                    self.assertEqual(
                        checkpoint["training_state"]["tokens_seen"], 2 * target
                    )
                    self.assertEqual(
                        checkpoint["training_state"]["next_batch_index"], target
                    )
                    self.assertEqual(len(completed), target)
                    self.train(directory, name, settings, resume_from=checkpoint_path)
                    final = load_checkpoint_file(run / "checkpoint_step_00000003.pt")
                    self.assert_payload_equal(final, expected_final)

    def test_stop_while_finishing_validation_saves_without_waiting_for_evaluator(self):
        pending = WaitingFuture()
        stop = Event()
        process_stops = []

        class PendingProcess:
            def __init__(self, *args):
                pass

            def submit(self, step, path):
                return pending

            def stop(self, **kwargs):
                process_stops.append(kwargs)

        errors = []
        with TemporaryDirectory() as directory:

            def train():
                try:
                    pretrain_model(
                        tiny_tokenizer(),
                        tiny_data(validation=True),
                        tiny_settings(max_validation_tokens=2),
                        tiny_config(),
                        directory,
                        "stopped",
                        device="cpu",
                        stop_requested=stop.is_set,
                    )
                except Exception as error:  # noqa: BLE001 -- Surface background test failures.
                    errors.append(error)

            with patch(
                "fraise.training.validation.runner.ValidationProcess", PendingProcess
            ):
                worker = Thread(target=train, daemon=True)
                worker.start()
                try:
                    self.assertTrue(
                        pending.waiting.wait(timeout=5),
                        "training never reached its validation wait",
                    )
                    self.assertFalse(pending.done())
                    stop.set()
                    worker.join(timeout=2)
                    self.assertFalse(
                        worker.is_alive(), "Stop waited for the pending evaluator"
                    )
                finally:
                    stop.set()
                    worker.join(timeout=2)
            self.assertEqual(errors, [])
            self.assertFalse(pending.done())
            self.assertTrue(process_stops)
            run = Path(directory) / "stopped"
            metadata = read_json(run / "model_meta.json")
            self.assertEqual(metadata["training_status"], "stopped")
            self.assertEqual(metadata["global_step"], 2)
            self.assertEqual(
                [result["status"] for result in metadata["validation_results"]],
                ["cancelled", "cancelled"],
            )
            checkpoint = load_checkpoint_file(run / metadata["latest_checkpoint"])
            self.assertEqual(checkpoint["training_state"]["tokens_seen"], 4)
            self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], 2)


class ValidationStopTests(unittest.TestCase):
    def test_stop_releases_both_queue_backpressure_and_final_validation_wait(self):
        for action in ("backpressure", "finish"):
            with self.subTest(action=action):
                pending = WaitingFuture()
                stop = Event()
                submitted, resets, errors = [], [], []

                def submit(step, path, *, submitted=submitted, pending=pending):
                    submitted.append(step)
                    return pending

                queue = ValidationQueue(
                    submit,
                    lambda message: None,
                    reset=lambda resets=resets: resets.append(True),
                    stop_requested=stop.is_set,
                )
                queue.queue(1, Path("one.pt"))
                queue.queue(2, Path("two.pt"))

                def wait(*, action=action, queue=queue, errors=errors):
                    try:
                        if action == "backpressure":
                            queue.queue(3, Path("three.pt"))
                        else:
                            queue.report(wait=True)
                    except Exception as error:  # noqa: BLE001 -- Surface background test failures.
                        errors.append(error)

                worker = Thread(target=wait, daemon=True)
                worker.start()
                try:
                    self.assertTrue(pending.waiting.wait(timeout=1))
                    stop.set()
                    worker.join(timeout=2)
                    self.assertFalse(
                        worker.is_alive(), "Stop waited for unresolved validation"
                    )
                finally:
                    stop.set()
                    worker.join(timeout=2)
                self.assertEqual(errors, [])
                self.assertFalse(pending.done())
                self.assertEqual(submitted, [1])
                self.assertEqual(resets, [True])
                self.assertEqual(queue.protected_paths(), set())
                self.assertEqual([result["step"] for result in queue.results], [1, 2])
                self.assertTrue(
                    all(result["status"] == "cancelled" for result in queue.results)
                )

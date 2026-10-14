import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from fixtures import tiny_config

from fraise.model.model import FraiseModel
from fraise.training.run.checkpoints import (
    CheckpointStore,
    checkpoint_to_model,
    load_checkpoint_file,
    load_model_weights,
)
from fraise.training.run.state import TrainingState


class CheckpointIOTests(unittest.TestCase):
    def test_exported_weights_match_and_do_not_overwrite_the_checkpoint(self):
        model = FraiseModel(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            store = CheckpointStore(directory)
            checkpoint = store.save(
                model, optimizer, scheduler, TrainingState(global_step=1)
            )
            before = checkpoint.read_bytes()
            output = checkpoint_to_model(checkpoint, directory / "weights.pt")
            self.assertEqual(checkpoint.read_bytes(), before)
            weights = load_model_weights(output)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(weights[key], value)
            with self.assertRaisesRegex(ValueError, "must differ"):
                checkpoint_to_model(checkpoint, checkpoint)

    def test_invalid_checkpoint_payload_fails_at_the_format_boundary(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            for payload in (
                {"schema_version": 99},
                {"schema_version": 3, "model_state_dict": {}},
                {
                    "schema_version": 3,
                    "model_state_dict": [],
                    "optimizer_state_dict": {},
                    "scheduler_state_dict": {},
                    "training_state": {},
                    "rng_state": {},
                },
            ):
                torch.save(payload, path)
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    load_checkpoint_file(path)

    def test_checkpoints_load_without_unpickling_arbitrary_objects(self):
        import random

        import numpy as np

        from fraise.training.run.rng import capture_rng_state, restore_rng_state

        model = FraiseModel(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
        with TemporaryDirectory() as directory:
            np.random.seed(5)
            np.random.standard_normal()  # populate the cached Gaussian
            random.seed(6)
            path = CheckpointStore(Path(directory)).save(
                model, optimizer, scheduler, TrainingState(global_step=1)
            )
            self.assertEqual(
                torch.serialization.get_unsafe_globals_in_checkpoint(path), []
            )
            expected = (np.random.standard_normal(3), random.random())
            np.random.seed(0)
            random.seed(0)
            restore_rng_state(load_checkpoint_file(path)["rng_state"])
            np.testing.assert_array_equal(np.random.standard_normal(3), expected[0])
            self.assertEqual(random.random(), expected[1])

            # Schema 3 stored NumPy's tuple state; it still restores exactly.
            legacy = torch.load(path, weights_only=True)
            np.random.seed(5)
            np.random.standard_normal()
            legacy["schema_version"] = 3
            legacy["rng_state"] = capture_rng_state() | {"numpy": np.random.get_state()}
            torch.save(legacy, path)
            expected = np.random.standard_normal(3)
            restore_rng_state(load_checkpoint_file(path)["rng_state"])
            np.testing.assert_array_equal(np.random.standard_normal(3), expected)

    def test_pickled_code_is_refused(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save({"weight": torch.zeros(1), "hook": Path("x")}, path)
            with self.assertRaisesRegex(ValueError, "untrusted"):
                load_model_weights(path)

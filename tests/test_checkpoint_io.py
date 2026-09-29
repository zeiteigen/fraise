import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from fixtures import tiny_config

from fraise.model.model import FraiseModel
from fraise.training.checkpoints import (
    CheckpointStore,
    checkpoint_to_model,
    load_checkpoint_file,
    load_model_weights,
)
from fraise.training.state import TrainingState


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

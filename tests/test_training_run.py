import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from fixtures import tiny_config, tiny_data, tiny_settings, tiny_tokenizer

from fraise.io import read_json
from fraise.model.model import FraiseModel
from fraise.training.training import pretrain_transformer


class TrainingRunTests(unittest.TestCase):
    def test_small_run_retains_latest_checkpoint_and_archives_inputs(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(),
                tiny_settings(checkpoint_keep_last=1),
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            run = Path(directory) / "tiny"
            self.assertEqual(
                [path.name for path in run.glob("checkpoint_step_*.pt")],
                ["checkpoint_step_00000002.pt"],
            )
            for name in (
                "tiny.pt",
                "model_config.json",
                "training_hyperparameters.json",
                "tokenizer.json",
                "training.log",
            ):
                self.assertTrue((run / name).is_file(), name)
            meta = read_json(run / "model_meta.json")
            self.assertTrue(meta["training_finished"])
            self.assertNotIn("all_messages", meta)

    def test_interrupted_resume_matches_uninterrupted_weights_and_schedule(self):
        settings = tiny_settings(checkpoint_keep_last=2)
        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            original = pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(),
                settings,
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            run = Path(directory) / "tiny"
            checkpoint = run / "checkpoint_step_00000001.pt"
            resumed = pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(),
                settings,
                tiny_config(),
                directory,
                "tiny",
                checkpoint,
                device="cpu",
            )
            for name, weights in original.state_dict().items():
                torch.testing.assert_close(
                    resumed.state_dict()[name], weights, rtol=0, atol=0
                )
            final = torch.load(run / "checkpoint_step_00000002.pt", weights_only=False)
            self.assertEqual(final["training_state"]["tokens_seen"], 4)
            self.assertEqual(final["scheduler_state_dict"]["last_epoch"], 2)

    def test_separate_cpu_validation_completes_and_is_recorded(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(validation=True),
                tiny_settings(max_validation_tokens=4),
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            meta = read_json(Path(directory) / "tiny" / "model_meta.json")
            self.assertIsNotNone(meta["validation"])
            self.assertEqual(meta["validation"][0], 2)
            self.assertEqual(meta["validation_failures"], [])

    def test_resume_across_shuffled_source_and_epoch_boundaries(self):
        from datasets import Dataset

        from fraise.data.types import TrainingData

        data = TrainingData(
            {
                "first": Dataset.from_dict({"text": ["a b c d e", "e d c b a"]}),
                "second": Dataset.from_dict({"text": ["a c e b d"]}),
            }
        )
        settings = tiny_settings(
            batch_size=2,
            minibatch_size=1,
            num_epochs=2,
            shuffle=True,
            dataset_token_counts={"first": 16, "second": 4},
            max_train_tokens=20,
            checkpoint_keep_last=8,
        )
        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            expected = pretrain_transformer(
                tiny_tokenizer(),
                data,
                settings,
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            for step in (1, 5):
                checkpoint = Path(directory) / "tiny" / f"checkpoint_step_{step:08d}.pt"
                resumed = pretrain_transformer(
                    tiny_tokenizer(),
                    data,
                    settings,
                    tiny_config(),
                    directory,
                    "tiny",
                    checkpoint,
                    device="cpu",
                )
                for name, weights in expected.state_dict().items():
                    torch.testing.assert_close(
                        resumed.state_dict()[name], weights, rtol=0, atol=0
                    )

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
    def test_zero_validation_budget_skips_tokenization_and_workers_for_both_backends(
        self,
    ):
        from datasets import Dataset

        data = tiny_data()
        # This supplied held-out split cannot be tokenized; disabled validation ignores it.
        data.validation = Dataset.from_dict({"unused": ["held-out text"]})
        for backend in ("cpu", "npu"):
            with (
                self.subTest(backend=backend),
                TemporaryDirectory() as directory,
                patch.object(FraiseModel, "compile", lambda self: None),
                patch("fraise.training.validation_runner.ValidationProcess") as worker,
            ):
                pretrain_transformer(
                    tiny_tokenizer(),
                    data,
                    tiny_settings(
                        max_validation_tokens=0, validation_device=backend, num_epochs=2
                    ),
                    tiny_config(),
                    directory,
                    "tiny",
                    device="cpu",
                )
                worker.assert_not_called()
                run = Path(directory) / "tiny"
                meta = read_json(run / "model_meta.json")
                self.assertTrue(meta["training_finished"])
                self.assertEqual(meta["tokens_seen"], 8)
                self.assertIsNone(meta["validation"])
                self.assertEqual(meta.get("validation_results", []), [])
                self.assertEqual(meta["validation_failures"], [])
                self.assertTrue((run / "tiny.pt").is_file())
                self.assertIn(
                    "Validation disabled: max_validation_tokens is 0.",
                    (run / "training.log").read_text(),
                )

    def test_three_epochs_each_consume_the_full_budget_and_validate(self):
        settings = tiny_settings(
            batch_size=2,
            minibatch_size=1,
            dataset_token_counts={"fixture": 5},
            max_train_tokens=5,
            num_epochs=3,
            checkpoint_interval_steps=1000,
            checkpoint_keep_last=3,
            max_validation_tokens=2,
        )
        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
        ):
            pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(validation=True),
                settings,
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            run = Path(directory) / "tiny"
            meta = read_json(run / "model_meta.json")
            self.assertTrue(meta["training_finished"])
            self.assertEqual(meta["tokens_seen"], 15)
            self.assertEqual(meta["global_step"], 6)
            self.assertEqual(meta["training_state"]["epoch"], 3)
            self.assertEqual(
                [result["step"] for result in meta["validation_results"]], [2, 4, 6]
            )
            log = (run / "training.log").read_text()
            self.assertIn("5 targets per epoch for 3 epochs (15 total)", log)
            for epoch, step in ((1, 2), (2, 4), (3, 6)):
                self.assertIn(f"Epoch {epoch}/3: loss", log)
                self.assertIn(f"tokens 5/5 this epoch, total {epoch * 5}/15", log)
                checkpoint = torch.load(
                    run / f"checkpoint_step_{step:08d}.pt", weights_only=False
                )
                self.assertEqual(checkpoint["training_state"]["tokens_seen"], epoch * 5)
                self.assertEqual(checkpoint["scheduler_state_dict"]["last_epoch"], step)

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
            self.assertEqual(
                [result["step"] for result in meta["validation_results"]], [1, 2]
            )
            self.assertTrue(
                all(
                    result["status"] == "completed"
                    for result in meta["validation_results"]
                )
            )
            log = (Path(directory) / "tiny" / "training.log").read_text()
            for message in (
                "worker ready",
                "Started checkpoint",
                "Loading checkpoint",
                "Progress 4/4",
                "Validation loss at global step 1",
                "Validation loss at global step 2",
            ):
                self.assertIn(message, log)

    def test_validation_outcomes_are_durable_before_finish_and_survive_resume(self):
        from fraise.training.artifacts import RunArtifacts

        original_finish = RunArtifacts.finish
        snapshots = []

        def inspect_finish(artifacts, *args, **kwargs):
            snapshots.append(read_json(artifacts.path)["validation_results"])
            return original_finish(artifacts, *args, **kwargs)

        with (
            TemporaryDirectory() as directory,
            patch.object(FraiseModel, "compile", lambda self: None),
            patch.object(RunArtifacts, "finish", inspect_finish),
        ):
            settings = tiny_settings(max_validation_tokens=4)
            pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(validation=True),
                settings,
                tiny_config(),
                directory,
                "tiny",
                device="cpu",
            )
            checkpoint = Path(directory) / "tiny" / "checkpoint_step_00000001.pt"
            pretrain_transformer(
                tiny_tokenizer(),
                tiny_data(validation=True),
                settings,
                tiny_config(),
                directory,
                "tiny",
                checkpoint,
                device="cpu",
            )
            self.assertEqual([record["step"] for record in snapshots[0]], [1, 2])
            self.assertEqual([record["step"] for record in snapshots[1]], [1, 2, 2])

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
            shuffle_per_epoch=True,
            shuffle_data=True,
            dataset_token_counts={"first": 11, "second": 4},
            max_train_tokens=15,
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
            final = torch.load(
                Path(directory) / "tiny" / "checkpoint_step_00000008.pt",
                weights_only=False,
            )
            self.assertEqual(final["training_state"]["tokens_seen"], 30)
            self.assertEqual(final["training_state"]["epoch"], 2)
            self.assertEqual(final["scheduler_state_dict"]["last_epoch"], 8)
            for step in (1, 4, 5, 8):
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

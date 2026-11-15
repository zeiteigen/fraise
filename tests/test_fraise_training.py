"""Macro-tokenized runs train, validate, stop, resume, and generate end to end."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

import torch
from datasets import Dataset
from fixtures import eager_loss, tiny_config, tiny_settings
from macro_fixtures import BYTE_TEXTS, byte_level_tokenizers, tiny_fraise_config

from fraise.data.types import TrainingData
from fraise.inference.artifacts import load_run
from fraise.inference.generation import generate
from fraise.inference.scoring import score_text
from fraise.io import read_json
from fraise.training.run.checkpoints import load_checkpoint_file
from fraise.training.runtime import TrainingRuntime
from fraise.training.training import pretrain_model

BUDGET = 120


def settings(**changes):
    return tiny_settings(
        **{
            "dataset_token_counts": {"fixture": BUDGET},
            "max_train_tokens": BUDGET,
            "sequence_length": 6,
            "batch_size": 4,
            "minibatch_size": 2,
            "checkpoint_interval_steps": 2,
            "checkpoint_keep_last": 20,
            "deterministic": True,
            "max_validation_tokens": 0,
        }
        | changes
    )


class FraiseTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.macro = byte_level_tokenizers()
        source = Dataset.from_dict({"text": BYTE_TEXTS})
        cls.data = TrainingData({"fixture": source}, source)

    def setUp(self):
        cache = TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        for active in (
            patch("datasets.config.HF_DATASETS_CACHE", Path(cache.name)),
            patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}),
            eager_loss(),
        ):
            active.start()
            self.addCleanup(active.stop)

    def train(self, directory, name, training, config=None, tokenizer=None, **options):
        return pretrain_model(
            tokenizer or self.macro,
            self.data,
            training,
            config or tiny_fraise_config(self.macro),
            directory,
            name,
            device="cpu",
            **options,
        )

    def test_fraise_run_counts_base_tokens_validates_and_generates(self):
        training = settings(max_validation_tokens=40, validation_timeout_seconds=60)
        with TemporaryDirectory() as directory:
            self.train(directory, "fraise", training)
            run = Path(directory) / "fraise"
            metadata = read_json(run / "model_meta.json")
            self.assertTrue(metadata["training_finished"])
            # Training ends with the macro reaching 120 base tokens.
            self.assertGreaterEqual(metadata["tokens_seen"], BUDGET)
            self.assertLess(
                metadata["tokens_seen"], BUDGET + self.macro.max_macro_length
            )
            completed = [
                result
                for result in metadata["validation_results"]
                if result["status"] == "completed"
            ]
            self.assertTrue(completed)
            self.assertGreater(completed[-1]["bits_per_byte"], 0)
            self.assertGreaterEqual(completed[-1]["base_tokens"], 40)
            self.assertIn("bits/byte", (run / "training.log").read_text())

            loaded = load_run(run, device="cpu")
            self.assertTrue(loaded.model.hierarchical)
            torch.testing.assert_close(
                loaded.model.macro_lengths,
                torch.as_tensor(self.macro.unit_lengths),
            )
            text, ids = generate(
                loaded.model, loaded.tokenizer, "The straw", max_new_tokens=6
            )
            self.assertTrue(text.startswith("The straw"))
            self.assertLessEqual(len(ids), len(self.base.encode("The straw").ids) + 6)
            score = score_text(loaded.model, loaded.tokenizer, "The strawberry field.")
            self.assertGreater(score["bits_per_byte"], 0)

    def test_stopped_fraise_run_resumes_exactly(self):
        training = settings()
        with TemporaryDirectory() as directory:
            self.train(directory, "reference", training)
            reference = Path(directory) / "reference"
            final = max(reference.glob("checkpoint_step_*.pt"))
            stop, completed = Event(), []
            original_step = TrainingRuntime.step

            def step_and_stop(runtime, batch):
                result = original_step(runtime, batch)
                completed.append(result)
                if len(completed) == 3:
                    stop.set()
                return result

            with patch.object(TrainingRuntime, "step", step_and_stop):
                self.train(directory, "resumed", training, stop_requested=stop.is_set)
            run = Path(directory) / "resumed"
            stopped = run / "checkpoint_step_00000003.pt"
            self.assertTrue(stopped.is_file())
            self.train(directory, "resumed", training, resume_from=stopped)
            expected = load_checkpoint_file(final)
            actual = load_checkpoint_file(run / final.name)
            for key in ("model_state_dict", "training_state"):
                self.assertEqual(actual[key].keys(), expected[key].keys())
                for name, value in expected[key].items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(
                            actual[key][name], value, rtol=0, atol=0
                        )
                    else:
                        self.assertEqual(actual[key][name], value, name)

    def test_symmetric_baseline_trains_on_the_same_base_token_budget(self):
        config = tiny_config(vocab_size=self.macro.get_vocab_size(), max_seq_len=8)
        with TemporaryDirectory() as directory:
            self.train(directory, "symmetric", settings(), config=config)
            symmetric = read_json(Path(directory) / "symmetric" / "model_meta.json")
            self.train(
                directory,
                "fraise",
                settings(),
            )
            fraise = read_json(Path(directory) / "fraise" / "model_meta.json")
            self.assertEqual(symmetric["tokens_seen"], fraise["tokens_seen"])
            self.train(
                directory,
                "base",
                settings(),
                config=tiny_config(
                    vocab_size=self.base.get_vocab_size(), max_seq_len=8
                ),
                tokenizer=self.base,
            )
            base = read_json(Path(directory) / "base" / "model_meta.json")
            self.assertEqual(base["tokens_seen"], BUDGET)


if __name__ == "__main__":
    unittest.main()

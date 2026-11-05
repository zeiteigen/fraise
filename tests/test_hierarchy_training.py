"""Hierarchical-embedding runs train, validate, resume, load, and generate."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from fixtures import eager_loss
from macro_fixtures import BYTE_TEXTS, byte_level_tokenizers, tiny_hierarchy_config
from test_fraise_training import settings

from fraise.data.types import TrainingData
from fraise.hierarchy import hierarchy_from_macro_tokenizer
from fraise.hierarchy.runs import run_hierarchy
from fraise.inference.artifacts import load_run
from fraise.inference.generation import generate
from fraise.inference.scoring import score_text
from fraise.io import read_json
from fraise.training.run.checkpoints import load_checkpoint_file
from fraise.training.runtime import TrainingRuntime
from fraise.training.training import pretrain_model

LEVELS = (280, 300, 320)


class HierarchyTrainingTests(unittest.TestCase):
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

    def config(self, **changes):
        return tiny_hierarchy_config(self.macro, LEVELS, [2, 3], **changes)

    def train(self, directory, name, training, config=None, **options):
        return pretrain_model(
            self.macro,
            self.data,
            training,
            config or self.config(),
            directory,
            name,
            device="cpu",
            **options,
        )

    def test_run_validates_loads_and_generates_from_saved_structure(self):
        training = settings(max_validation_tokens=40, validation_timeout_seconds=60)
        with TemporaryDirectory() as directory:
            definition = hierarchy_from_macro_tokenizer(self.macro, LEVELS, [2, 3])
            path = Path(directory) / "hierarchy.npz"
            sha256 = definition.save(path)
            levels = definition.config_levels()
            config = self.config(
                hierarchy={"levels": levels, "source": "file", "path": str(path), "sha256": sha256}
            )
            self.train(directory, "hierarchy", training, config=config)
            path.unlink()  # Inference must not need the hierarchy file.
            run = Path(directory) / "hierarchy"
            metadata = read_json(run / "model_meta.json")
            self.assertTrue(metadata["training_finished"])
            completed = [
                r for r in metadata["validation_results"] if r["status"] == "completed"
            ]
            self.assertTrue(completed)
            self.assertGreater(completed[-1]["bits_per_byte"], 0)

            loaded = load_run(run, device="cpu")
            recovered = loaded.model.embedding.definition()
            for a, b in zip(recovered.compositions, definition.compositions):
                np.testing.assert_array_equal(a, b)
            text, _ = generate(loaded.model, loaded.tokenizer, "The straw", max_new_tokens=6)
            self.assertTrue(text.startswith("The straw"))
            self.assertGreater(
                score_text(loaded.model, loaded.tokenizer, "The strawberry.")["bits_per_byte"], 0
            )
            _, from_run, _ = run_hierarchy(run)
            np.testing.assert_array_equal(
                from_run.residual_widths, definition.residual_widths
            )
            # Trained residuals moved away from their initialization.
            self.assertTrue(
                all(level.values.abs().sum() > 0 for level in loaded.model.embedding.levels)
            )

    def test_stopped_hierarchical_run_resumes_exactly(self):
        training = settings(num_epochs=1)
        config = self.config(hierarchy={"composition": "shared"})
        with TemporaryDirectory() as directory:
            self.train(directory, "reference", training, config=config)
            final = max((Path(directory) / "reference").glob("checkpoint_step_*.pt"))
            stop, completed = Event(), []
            original_step = TrainingRuntime.step

            def step_and_stop(runtime, batch):
                result = original_step(runtime, batch)
                completed.append(result)
                if len(completed) == 2:
                    stop.set()
                return result

            with patch.object(TrainingRuntime, "step", step_and_stop):
                self.train(
                    directory, "resumed", training, config=config, stop_requested=stop.is_set
                )
            run = Path(directory) / "resumed"
            stopped = run / "checkpoint_step_00000002.pt"
            self.train(directory, "resumed", training, config=config, resume_from=stopped)
            expected = load_checkpoint_file(final)["model_state_dict"]
            actual = load_checkpoint_file(run / final.name)["model_state_dict"]
            self.assertEqual(actual.keys(), expected.keys())
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "model_config"):
                self.train(
                    directory, "resumed", training, config=self.config(), resume_from=stopped
                )


if __name__ == "__main__":
    unittest.main()

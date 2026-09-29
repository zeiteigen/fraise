import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset
from fixtures import tiny_config, tiny_settings, tiny_tokenizer

from fraise.config import load_model_config, load_training_settings
from fraise.data import loading
from fraise.data.sources import SOURCE_NAMES
from fraise.data.types import TrainingData
from fraise.io import write_json


class DatasetSelectionTests(unittest.TestCase):
    def test_active_source_limits_and_independent_validation_are_explicit(self):
        train = Dataset.from_dict({"text": ["a"]})
        validation = Dataset.from_dict({"text": ["b"]})
        counts = {name: 0 for name in SOURCE_NAMES}
        counts["cosmopedia-khanacademy"] = 7

        def fake_stream(source, directory, **kwargs):
            if source.name == "fineweb-edu":
                self.assertFalse(kwargs["include_train"])
                return TrainingData(validation=validation)
            self.assertEqual(kwargs["max_train_tokens"], 7)
            return TrainingData({source.name: train})

        with (
            TemporaryDirectory() as root,
            patch.object(loading, "load_stream", side_effect=fake_stream),
            patch.object(loading, "load_babylm") as baby,
        ):
            result = loading.load_training_data(
                counts, tiny_tokenizer(), root=Path(root)
            )
        baby.assert_not_called()
        self.assertIs(result.train["cosmopedia-khanacademy"], train)
        self.assertIs(result.validation, validation)

    def test_preview_skips_validation_loading(self):
        counts = {"babylm": 4}
        with (
            TemporaryDirectory() as root,
            patch.object(
                loading, "load_babylm", return_value=Dataset.from_dict({"text": ["a"]})
            ),
            patch.object(loading, "load_stream") as stream,
        ):
            result = loading.load_training_data(
                counts, tiny_tokenizer(), root=Path(root), validation=False
            )
        stream.assert_not_called()
        self.assertIsNone(result.validation)

    def test_json_and_direct_configuration_have_identical_defaults(self):
        with TemporaryDirectory() as directory:
            model, training = (
                Path(directory) / "model.json",
                Path(directory) / "training.json",
            )
            write_json(model, asdict(tiny_config()))
            write_json(training, asdict(tiny_settings()))
            self.assertEqual(load_model_config(model), tiny_config())
            self.assertEqual(load_training_settings(training), tiny_settings())
            invalid = asdict(tiny_config()) | {"qk_dim": 3}
            write_json(model, invalid)
            with self.assertRaisesRegex(ValueError, "qk_dim"):
                load_model_config(model)

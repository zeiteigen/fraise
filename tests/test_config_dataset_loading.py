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
        ):
            result = loading.load_training_data(
                counts, tiny_tokenizer(), root=Path(root)
            )
        self.assertNotIn("babylm", SOURCE_NAMES)
        self.assertEqual(result.train.keys(), {"cosmopedia-khanacademy"})
        self.assertIs(result.train["cosmopedia-khanacademy"], train)
        self.assertIs(result.validation, validation)

    def test_tokenizer_sources_skip_the_fineweb_validation_prefix(self):
        calls = {}

        def fake_stream(source, directory, **kwargs):
            calls[source.name] = kwargs
            return TrainingData({source.name: Dataset.from_dict({"text": ["a"]})})

        with (
            TemporaryDirectory() as root,
            patch.object(loading, "load_stream", side_effect=fake_stream),
        ):
            loading.load_tokenizer_sources(
                ["fineweb-edu", "cosmopedia-stories"], root=Path(root), size_gb=0.1
            )
        fineweb = calls["fineweb-edu"]
        self.assertEqual(fineweb["validation_size_gb"], loading.VALIDATION_SIZE_GB)
        self.assertFalse(fineweb["include_validation"])
        self.assertAlmostEqual(fineweb["size_gb"], 0.1 + loading.VALIDATION_SIZE_GB)
        self.assertEqual(calls["cosmopedia-stories"]["validation_size_gb"], 0)
        self.assertEqual(calls["cosmopedia-stories"]["size_gb"], 0.1)

    def test_preview_skips_validation_loading(self):
        counts = {"cosmopedia-stories": 4}
        with (
            TemporaryDirectory() as root,
            patch.object(
                loading,
                "load_stream",
                return_value=TrainingData(
                    {"cosmopedia-stories": Dataset.from_dict({"text": ["a"]})}
                ),
            ) as stream,
        ):
            result = loading.load_training_data(
                counts, tiny_tokenizer(), root=Path(root), validation=False
            )
        self.assertEqual(
            [call.args[0].name for call in stream.call_args_list],
            ["cosmopedia-stories"],
        )
        self.assertFalse(stream.call_args.kwargs["include_validation"])
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

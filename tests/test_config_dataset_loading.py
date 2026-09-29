import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from datasets_mgr import DATASET_NAME_COLUMN, get_dataset_sources  # noqa: E402


class DatasetSelectionTests(unittest.TestCase):
    def test_training_loader_uses_token_limit_and_source_map(self):
        calls = []
        training = Dataset.from_dict({"text": ["training"]})
        training.info.dataset_name = config.KHAN_NAME
        validation = Dataset.from_dict({"text": ["validation"]})
        validation.info.dataset_name = config.FINEWEB_NAME

        def khan(**kwargs):
            calls.append(("khan", kwargs))
            return training

        def fineweb(*, include_train=True):
            self.assertFalse(include_train)
            return DatasetDict({"validation": validation})

        loaders = dict(config.DATASET_LOADERS)
        loaders[config.KHAN_NAME] = khan
        loaders[config.FINEWEB_NAME] = fineweb
        tokenizer = object()
        with patch.object(config, "DATASET_LOADERS", loaders):
            combined = config.get_dataset_loader(
                {config.KHAN_NAME},
                tokenizer=tokenizer,
                dataset_token_counts={config.KHAN_NAME: 7},
            )

        self.assertEqual(
            calls,
            [("khan", {"max_train_tokens": 7, "tokenizer": tokenizer})],
        )
        self.assertNotIn(DATASET_NAME_COLUMN, combined["train"].column_names)
        self.assertIs(get_dataset_sources(combined)[config.KHAN_NAME], training)

    def test_model_config_rejects_invalid_dimensions_before_loading_data(self):
        valid = {
            "vocab_size": 8,
            "residual_stream_dim": 8,
            "head_dim": 4,
            "qk_dim": 4,
            "num_heads": 1,
            "swiglu_up_dim": 16,
            "num_layers": 1,
            "max_seq_len": 4,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            invalid_cases = (
                ({"num_layers": 0}, "num_layers"),
                ({"qk_dim": 3}, "qk_dim"),
            )
            for changed, message in invalid_cases:
                with self.subTest(changed=changed):
                    path.write_text(json.dumps(valid | changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        config.load_model_config(path)

    def test_only_active_training_source_is_loaded(self):
        calls = []

        def source(name):
            def load():
                calls.append(name)
                dataset = Dataset.from_dict({"text": [name]})
                dataset.info.dataset_name = name
                return dataset
            return load

        def fineweb(*, include_train=True):
            self.assertFalse(include_train)
            validation = Dataset.from_dict({"text": ["validation"]})
            validation.info.dataset_name = config.FINEWEB_NAME
            calls.append("fineweb-validation")
            return DatasetDict({"validation": validation})

        loaders = {name: source(name) for name in config.DATASET_LOADERS}
        loaders[config.FINEWEB_NAME] = fineweb
        with patch.object(config, "DATASET_LOADERS", loaders):
            dataset = config.get_dataset_loader({config.BABYLM_NAME})

        self.assertEqual(calls, [config.BABYLM_NAME, "fineweb-validation"])
        self.assertEqual(dataset["train"]["text"], [config.BABYLM_NAME])
        self.assertEqual(dataset["validation"]["text"], ["validation"])


if __name__ == "__main__":
    unittest.main()

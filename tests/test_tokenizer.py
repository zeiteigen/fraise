import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

import tokenizer  # noqa: E402


class TokenizerCacheTests(unittest.TestCase):
    def test_budgeted_text_batches_stop_at_byte_limit(self):
        source = Dataset.from_dict({"text": ["é", "abc", "later"]})
        batches = list(tokenizer._iter_text_batches(source, batch_size=3, max_bytes=3))
        self.assertEqual(batches, [["é", "abc"]])

    def test_budgeted_tokenizer_training_uses_each_source_limit(self):
        first = Dataset.from_dict({"text": ["a" * 10, "unused first"]})
        first.info.dataset_name = "first"
        second = Dataset.from_dict({"text": ["b" * 10, "unused second"]})
        second.info.dataset_name = "second"
        from datasets_mgr import unified_dataset_loader

        combined = unified_dataset_loader([first, second])
        seen = []
        original = tokenizer._iter_text_batches

        def record_batches(source, **kwargs):
            for batch in original(source, **kwargs):
                seen.extend(batch)
                yield batch

        with patch.object(tokenizer, "_iter_text_batches", side_effect=record_batches):
            tokenizer._train_tokenizer(
                combined, 32, {"first": 1, "second": 0}
            )
        self.assertEqual(seen, ["a" * 10])

    def test_existing_tokenizer_file_does_not_retrain(self):
        with TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            dataset = Dataset.from_dict({"text": ["test document"]})
            dataset.info.dataset_name = "fixture"

            persisted = Tokenizer(BPE())
            persisted.add_special_tokens(tokenizer.SPECIAL_TOKENS)

            with patch.object(tokenizer, "TOKENIZER_DIR", cache_dir), patch.object(
                tokenizer,
                "TOKENIZER_METADATA_FILE",
                cache_dir / "metadata.json",
            ):
                cache_path = tokenizer._tokenizer_path(32, "fixture")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                persisted.save(str(cache_path))
                tokenizer._write_metadata(
                    [
                        {
                            "schema_version": tokenizer.TOKENIZER_SCHEMA_VERSION,
                            "path": cache_path.name,
                            "vocab_size": 32,
                            "dataset": "fixture",
                            "dataset_size_bytes": tokenizer.get_dataset_size_bytes(
                                dataset
                            ),
                            "dataset_fingerprint": tokenizer.get_dataset_fingerprint(
                                dataset
                            ),
                            "training_split": "train",
                            "tokenizer_sha256": tokenizer._sha256_file(cache_path),
                        }
                    ]
                )

                with patch.object(
                    tokenizer,
                    "_train_tokenizer",
                    side_effect=AssertionError("tokenizer was retrained"),
                ):
                    loaded = tokenizer.get_or_train_tokenizer(32, dataset)

            self.assertEqual(loaded.get_vocab_size(), persisted.get_vocab_size())

    def test_training_metadata_uses_train_source_size(self):
        with TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            train = Dataset.from_dict({"text": ["training document"]})
            train.info.dataset_name = "fixture"
            train.info.download_size = 123
            validation = Dataset.from_dict({"text": ["validation document"]})
            validation.info.dataset_name = "fixture"
            validation.info.download_size = 456
            dataset = DatasetDict({"train": train, "validation": validation})

            with patch.object(tokenizer, "TOKENIZER_DIR", cache_dir), patch.object(
                tokenizer,
                "TOKENIZER_METADATA_FILE",
                cache_dir / "metadata.json",
            ):
                tokenizer.get_or_train_tokenizer(32, dataset)
                with (cache_dir / "metadata.json").open(
                    encoding="utf-8"
                ) as metadata_file:
                    metadata = json.load(metadata_file)

            self.assertEqual(metadata[0]["dataset_size_bytes"], 123)


if __name__ == "__main__":
    unittest.main()

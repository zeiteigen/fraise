import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset
from fixtures import tiny_tokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from fraise.io import write_json
from fraise.tokenizer import artifacts, loading
from fraise.tokenizer.migration import migrate_metadata
from fraise.tokenizer.training import iter_text_batches, train_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_explicit_loading_preserves_saved_behavior(self):
        tokenizer = Tokenizer(
            WordLevel(vocab=tiny_tokenizer().get_vocab(), unk_token="<|pad|>")
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(str(path))
            loaded = loading.load_tokenizer(path)
            self.assertEqual(loaded.to_str(), tokenizer.to_str())
            self.assertEqual(loaded.encode("a").ids, [3])

    def test_missing_special_tokens_are_rejected_without_repair(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            Tokenizer(WordLevel(vocab={"a": 0})).save(str(path))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "missing required token"):
                loading.load_tokenizer(path)
            self.assertEqual(path.read_bytes(), before)

    def test_bounded_training_text_keeps_complete_documents(self):
        source = Dataset.from_dict({"text": ["é", "abc", "later"]})
        self.assertEqual(
            list(iter_text_batches(source, batch_size=3, max_bytes=3)), [["é", "abc"]]
        )
        with self.assertRaises(ValueError):
            train_tokenizer({"fixture": source}, 32)

    def test_artifacts_reuse_matching_requests_and_preserve_other_corpora(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            sources = {"fixture": Dataset.from_dict({"text": ["a b c d e"]})}
            first = artifacts.create_tokenizer(sources, 300, directory)
            before = first.read_bytes()
            with patch.object(
                artifacts, "train_tokenizer", side_effect=AssertionError("retrained")
            ):
                self.assertEqual(
                    artifacts.create_tokenizer(sources, 300, directory), first
                )
            second = artifacts.create_tokenizer(
                {"fixture": Dataset.from_dict({"text": ["changed corpus"]})},
                300,
                directory,
            )
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(len(artifacts.read_metadata(directory)), 2)

    def test_legacy_metadata_has_an_explicit_migration(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            tiny_tokenizer().save(str(directory / "old.json"))
            write_json(
                directory / "metadata.json",
                {"tokenizers": [{"schema_version": 2, "path": "old.json"}]},
            )
            with self.assertRaisesRegex(ValueError, "migrate-metadata"):
                artifacts.read_metadata(directory)
            self.assertEqual(migrate_metadata(directory), 1)
            self.assertEqual(artifacts.read_metadata(directory)[0]["schema_version"], 3)
            backup = directory / "metadata.v2.json"
            before = backup.read_bytes()
            migrate_metadata(directory)
            self.assertEqual(backup.read_bytes(), before)

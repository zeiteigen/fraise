import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_dataset as real_load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from datasets_mgr import hf_dataset  # noqa: E402
from datasets_mgr.finewebedu import use_fineweb_edu  # noqa: E402


class HFDatasetTests(unittest.TestCase):
    def test_token_limit_stops_stream_after_training_prefix(self):
        tokenizer = Tokenizer(
            WordLevel(
                vocab={
                    "[UNK]": 0,
                    "[BOS]": 1,
                    "[EOS]": 2,
                    "hold": 3,
                    "alpha": 4,
                    "beta": 5,
                    "gamma": 6,
                    "unused": 7,
                },
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )
        rows = [
            {"text": "hold"},
            {"text": "alpha beta"},
            {"text": "gamma"},
            {"text": "unused"},
        ]
        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/token-limited",
                data_dir=directory,
            )
            with patch.object(
                hf_dataset.HFDataset, "_stream_source", return_value=iter(rows)
            ):
                result = source.load(
                    validation_size_gb=4e-9,
                    max_train_tokens=6,
                    tokenizer=tokenizer,
                )

            self.assertEqual(result["validation"]["text"], ["hold"])
            self.assertEqual(result["train"]["text"], ["alpha beta", "gamma"])
            with patch.object(
                hf_dataset.HFDataset,
                "_stream_source",
                side_effect=AssertionError("token-limited source was re-read"),
            ):
                cached = source.load(
                    validation_size_gb=4e-9,
                    max_train_tokens=6,
                    tokenizer=tokenizer,
                )
            self.assertEqual(cached["train"]["text"], ["alpha beta", "gamma"])

    def test_token_limit_requires_tokenizer(self):
        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/token-limited",
                data_dir=directory,
            )
            with self.assertRaisesRegex(ValueError, "tokenizer is required"):
                source.download(max_train_tokens=1)

    def test_fineweb_validation_only_download_is_bounded_to_validation_size(self):
        with patch.object(hf_dataset.HFDataset, "load") as load:
            use_fineweb_edu(include_train=False)
        self.assertEqual(load.call_args.kwargs["size_gb"], 0.05)
        self.assertEqual(load.call_args.kwargs["validation_size_gb"], 0.05)
        self.assertFalse(load.call_args.kwargs["include_train"])

    def test_validation_only_rejects_missing_validation_before_download(self):
        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/no-validation", data_dir=directory
            )
            with patch.object(hf_dataset.HFDataset, "download") as download:
                with self.assertRaisesRegex(ValueError, "validation split"):
                    source.load(include_train=False)
                download.assert_not_called()

    def test_stream_download_is_bounded_and_split(self):
        rows = [
            {"text": "abc"},
            {"text": "defgh"},
            {"text": "ijk"},
        ]

        def fake_load_dataset(path, *args, **kwargs):
            if kwargs.get("streaming"):
                return iter(rows)
            return real_load_dataset(path, *args, **kwargs)

        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/stream",
                data_dir=directory,
                dataset_name="fixture-stream",
                cache_prefix="fixture_stream",
            )
            with patch.object(
                hf_dataset,
                "load_dataset",
                side_effect=fake_load_dataset,
            ) as load_mock:
                result = source.load(
                    size_gb=6e-9,
                    validation_size_gb=3e-9,
                )

            self.assertIsInstance(result, DatasetDict)
            self.assertEqual(result["validation"]["text"], ["abc"])
            self.assertEqual(result["train"]["text"], ["defgh"])
            self.assertEqual(
                result["validation"].info.dataset_name,
                "fixture-stream",
            )
            self.assertEqual(load_mock.call_count, 2)

            # The manifest is keyed by the requested size, so this second
            # load does not invoke the remote streaming source again.
            with patch.object(
                hf_dataset.HFDataset,
                "_stream_source",
                side_effect=AssertionError("remote source was re-read"),
            ):
                cached = source.load(
                    size_gb=6e-9,
                    validation_size_gb=3e-9,
                )
            self.assertEqual(cached["train"]["text"], ["defgh"])

            validation_only = source.load(
                size_gb=6e-9,
                validation_size_gb=3e-9,
                include_train=False,
            )
            self.assertEqual(list(validation_only), ["validation"])
            self.assertEqual(validation_only["validation"]["text"], ["abc"])

    def test_complete_text_file_source_uses_snapshot_download(self):
        def fake_snapshot_download(**kwargs):
            local_dir = Path(kwargs["local_dir"])
            (local_dir / "one.train.txt").write_text(
                "one\ntwo\n", encoding="utf-8"
            )
            return str(local_dir)

        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/files",
                revision="fixture-revision",
                data_dir=directory,
                dataset_name="fixture-files",
                source_kind="text_files",
                file_patterns=("*.train.txt",),
            )
            with patch.object(
                hf_dataset,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as snapshot_mock:
                result = source.load()

            self.assertIsInstance(result, Dataset)
            self.assertEqual(result["text"], ["one", "two"])
            self.assertEqual(result.info.dataset_name, "fixture-files")
            snapshot_mock.assert_called_once()

    def test_arrow_cache_skips_loading_and_invalidates_on_file_change(self):
        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/cache",
                data_dir=directory,
                dataset_name="fixture-cache",
            )
            with patch.object(
                hf_dataset.HFDataset,
                "_stream_source",
                return_value=iter([{"text": "old"}]),
            ):
                first = source.load(size_gb=1e-8)

            with patch.object(
                hf_dataset,
                "load_dataset",
                side_effect=AssertionError("Arrow was regenerated"),
            ):
                second = hf_dataset.HFDataset(
                    repo_id="fixture/cache",
                    data_dir=directory,
                    dataset_name="fixture-cache",
                ).load(size_gb=1e-8)
            self.assertEqual(second["text"], ["old"])
            self.assertEqual(second._fingerprint, first._fingerprint)

            source._output_path("train").write_text(
                '{"text":"new"}\n', encoding="utf-8"
            )
            with patch.object(
                hf_dataset.HFDataset,
                "_stream_source",
                side_effect=AssertionError("source was re-downloaded"),
            ):
                changed = source.load(size_gb=1e-8)
            self.assertEqual(changed["text"], ["new"])

    def test_partial_arrow_cache_only_reparses_missing_split(self):
        with TemporaryDirectory() as directory:
            source = hf_dataset.HFDataset(
                repo_id="fixture/partial",
                data_dir=directory,
                dataset_name="fixture-partial",
            )

            def fake_load_dataset(path, *args, **kwargs):
                if kwargs.get("streaming"):
                    return iter([{"text": "abc"}, {"text": "def"}])
                return real_load_dataset(path, *args, **kwargs)

            with patch.object(hf_dataset, "load_dataset", side_effect=fake_load_dataset):
                source.load(
                    size_gb=6e-9,
                    validation_size_gb=3e-9,
                    include_train=False,
                )
            with patch.object(
                hf_dataset, "load_dataset", side_effect=fake_load_dataset
            ) as loader:
                complete = source.load(
                    size_gb=6e-9,
                    validation_size_gb=3e-9,
                )
            self.assertEqual(complete["train"]["text"], ["def"])
            self.assertEqual(complete["validation"]["text"], ["abc"])
            self.assertEqual(
                set(loader.call_args.kwargs["data_files"]),
                {"train"},
            )


if __name__ == "__main__":
    unittest.main()

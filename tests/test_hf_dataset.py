import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_dataset as real_load_dataset


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from datasets_mgr import hf_dataset  # noqa: E402


class HFDatasetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

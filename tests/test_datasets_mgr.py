import sys
from pathlib import Path
import unittest

from datasets import Dataset, DatasetDict


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from datasets_mgr import get_dataset_size_bytes, unified_dataset_loader  # noqa: E402


def _annotate(dataset: Dataset, name: str, download_size: int) -> Dataset:
    dataset.info.dataset_name = name
    dataset.info.download_size = download_size
    dataset.info.dataset_size = int(dataset.data.nbytes)
    dataset.info.size_in_bytes = download_size + dataset.info.dataset_size
    return dataset


class DatasetSizeTests(unittest.TestCase):
    def test_source_size_does_not_include_arrow_storage(self):
        dataset = _annotate(
            Dataset.from_dict({"text": ["example"]}),
            "fixture",
            123,
        )

        self.assertEqual(get_dataset_size_bytes(dataset), 123)

    def test_combined_splits_keep_their_own_source_sizes(self):
        source = DatasetDict(
            {
                "train": _annotate(
                    Dataset.from_dict({"text": ["training"]}),
                    "fixture",
                    100,
                ),
                "validation": _annotate(
                    Dataset.from_dict({"text": ["validation"]}),
                    "fixture",
                    10,
                ),
            }
        )

        combined = unified_dataset_loader([source])

        self.assertEqual(get_dataset_size_bytes(combined["train"]), 100)
        self.assertEqual(get_dataset_size_bytes(combined["validation"]), 10)


if __name__ == "__main__":
    unittest.main()

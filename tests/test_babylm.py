import sys
from pathlib import Path
import unittest
from unittest.mock import patch

from datasets import Dataset


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from datasets_mgr import babylm  # noqa: E402


class BabyLMChunkTests(unittest.TestCase):
    def test_chunks_preserve_newlines_and_exact_word_boundaries(self):
        rows = ["one  two", "three four five\nsix", "", "seven eight"]
        chunks = list(babylm._chunk_text_rows(rows, 3))

        self.assertEqual([chunk["text"] for chunk in chunks], [
            "one  two\nthree",
            " four five\nsix",
            "\n\nseven eight",
        ])
        self.assertEqual(
            "".join(chunk["text"] for chunk in chunks),
            "\n".join(rows),
        )

    def test_loader_retains_source_metadata(self):
        source = Dataset.from_dict({"text": ["one two", "three four five"]})
        source.info.dataset_name = babylm.DATASET_NAME
        source.info.download_size = 42

        with patch.object(babylm.HFDataset, "load", return_value=source):
            result = babylm.use_babylm(words_per_chunk=2)

        self.assertEqual(result["text"], ["one two", "\nthree four", " five"])
        self.assertEqual(result.info.dataset_name, babylm.DATASET_NAME)
        self.assertEqual(result.info.download_size, 42)
        self.assertEqual(result.info.dataset_size, result.data.nbytes)

    def test_trailing_blank_lines_stay_with_last_chunk(self):
        chunks = list(babylm._chunk_text_rows(["one two ", "", ""], 2))
        self.assertEqual(chunks, [{"text": "one two \n\n"}])

    def test_invalid_chunk_size_is_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                babylm.use_babylm(words_per_chunk=value)


if __name__ == "__main__":
    unittest.main()

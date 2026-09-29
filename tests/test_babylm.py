import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datasets import Dataset

from fraise.data import babylm


class BabyLMChunkTests(unittest.TestCase):
    def test_chunks_preserve_newlines_and_exact_word_boundaries(self):
        rows = ["one  two", "three four five\nsix", "", "seven eight"]
        chunks = list(babylm._chunk_text_rows(rows, 3))

        self.assertEqual(
            [chunk["text"] for chunk in chunks],
            [
                "one  two\nthree",
                " four five\nsix",
                "\n\nseven eight",
            ],
        )
        self.assertEqual(
            "".join(chunk["text"] for chunk in chunks),
            "\n".join(rows),
        )

    def test_loader_retains_source_metadata(self):
        source = Dataset.from_dict({"text": ["one two", "three four five"]})
        source.info.dataset_name = babylm.DATASET_NAME
        source.info.download_size = 42

        with TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "one.train.txt").write_bytes(b"a" * 42)
            with patch("datasets.load_dataset", return_value=source):
                result = babylm.load_babylm(directory, words_per_chunk=2)

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
                babylm.load_babylm(Path("unused"), words_per_chunk=value)


if __name__ == "__main__":
    unittest.main()

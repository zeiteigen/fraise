import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

import tokenizer  # noqa: E402


class TokenizerCacheTests(unittest.TestCase):
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

                with patch.object(
                    tokenizer,
                    "_train_tokenizer",
                    side_effect=AssertionError("tokenizer was retrained"),
                ):
                    loaded = tokenizer.load_tokenizer(32, dataset)

            self.assertEqual(loaded.get_vocab_size(), persisted.get_vocab_size())


if __name__ == "__main__":
    unittest.main()

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_model_config  # noqa: E402


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "vocab_size": 8,
            "residual_stream_dim": 8,
            "head_dim": 4,
            "qk_dim": 4,
            "num_heads": 1,
            "swiglu_up_dim": 16,
            "num_layers": 1,
            "max_seq_len": 4,
        }
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.json"

    def write_config(self, option):
        self.path.write_text(json.dumps(self.config | option), encoding="utf-8")

    def test_weight_tying_is_optional_and_accepts_both_booleans(self):
        for option in ({}, {"tie_word_embeddings": False}, {"tie_word_embeddings": True}):
            with self.subTest(option=option):
                self.write_config(option)
                self.assertEqual(load_model_config(self.path), self.config | option)

    def test_weight_tying_rejects_non_boolean_values(self):
        for value in (0, 1, "true", "false", None, [], {}):
            with self.subTest(value=value):
                self.write_config({"tie_word_embeddings": value})
                with self.assertRaisesRegex(TypeError, "tie_word_embeddings"):
                    load_model_config(self.path)


if __name__ == "__main__":
    unittest.main()

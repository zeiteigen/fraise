import sys
from dataclasses import replace
from pathlib import Path
import unittest

from datasets import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets_mgr import unified_dataset_loader  # noqa: E402
from scripts.visualize_training_data import collect_examples, render_html  # noqa: E402
from train.hyperparameters import _validate_hyperparameters  # noqa: E402


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def encode_batch(self, texts, *, add_special_tokens):
        return [_Encoding([ord(char) for char in text]) for text in texts]

    def token_to_id(self, token):
        return 0 if token == "<|pad|>" else None

    def decode(self, ids, *, skip_special_tokens):
        return "".join(chr(token_id) for token_id in ids if token_id)


class VisualizeTrainingDataTests(unittest.TestCase):
    def test_windows_follow_source_budgets_stride_and_batch_boundaries(self):
        first = Dataset.from_dict({"text": ["abcdefgh"]})
        first.info.dataset_name = "first"
        second = Dataset.from_dict({"text": ["ijklmnop"]})
        second.info.dataset_name = "second"
        dataset = unified_dataset_loader([first, second], source_column=False)
        settings = _validate_hyperparameters(
            {
                "batch_size": 3,
                "learning_rate": 1e-3,
                "num_epochs": 1,
                "max_train_tokens": 14,
                "dataset_token_counts": {"first": 7, "second": 7},
                "sequence_length": 4,
                "packing_stride": 2,
                "shuffle": True,
            },
            {"max_seq_len": 4},
            {"first", "second"},
        )

        examples = collect_examples(dataset, _Tokenizer(), settings, 4)

        self.assertEqual([item["source"] for item in examples],
                         ["first", "first", "second", "second"])
        self.assertEqual([item["input_text"] for item in examples],
                         ["abcd", "cdef", "ijkl", "klmn"])
        self.assertEqual([item["target_text"] for item in examples],
                         ["bcde", "def", "jklm", "lmn"])
        self.assertEqual([item["valid_targets"] for item in examples], [4, 3, 4, 3])
        self.assertEqual([item["batch"] for item in examples], [0, 0, 0, 1])
        self.assertEqual(examples[1]["target_mask"], [True, True, True, False])
        self.assertEqual(examples[1]["input_ids"], [99, 100, 101, 102])
        self.assertIn("[99, 100, 101, 102]", render_html(examples, settings))

        first_only = unified_dataset_loader([first], source_column=False)
        first_settings = replace(
            settings, dataset_token_counts={"first": 7}, max_train_tokens=7
        )
        self.assertEqual(
            collect_examples(
                first_only, _Tokenizer(), first_settings, 2,
                first_source_only=True,
            ),
            examples[:2],
        )


if __name__ == "__main__":
    unittest.main()

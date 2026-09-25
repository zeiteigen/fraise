import sys
from pathlib import Path
import unittest

from datasets import Dataset


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from datasets_mgr import (  # noqa: E402
    DATASET_NAME_COLUMN,
    get_dataset_sources,
    unified_dataset_loader,
)
from train.hyperparameters import (  # noqa: E402
    _validate_hyperparameters,
    allocate_dataset_token_budgets,
)
from train.packing import (  # noqa: E402
    iter_dataset_packed_batches,
    pretokenize_split,
)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _LengthTokenizer:
    def encode_batch(self, texts, *, add_special_tokens):
        del add_special_tokens
        return [_Encoding(list(range(1, len(text) + 1))) for text in texts]


class TrainingDataMixTests(unittest.TestCase):
    def test_sources_are_retained_when_datasets_are_combined(self):
        first = Dataset.from_dict({"text": ["first"]})
        first.info.dataset_name = "first"
        second = Dataset.from_dict({"text": ["second", "third"]})
        second.info.dataset_name = "second"

        combined = unified_dataset_loader([first, second])
        sources = get_dataset_sources(combined)

        self.assertEqual(set(sources), {"first", "second"})
        self.assertEqual(sources["first"]["text"], ["first"])
        self.assertEqual(sources["second"]["text"], ["second", "third"])
        self.assertIn(DATASET_NAME_COLUMN, combined["train"].column_names)

    def test_percentage_validation_checks_names_and_sum(self):
        base = {
            "batch_size": 1,
            "learning_rate": 1e-3,
            "max_train_tokens": 10,
            "num_epochs": 1,
            "dataset_token_percentages": {"first": 25, "second": 75},
        }
        settings = _validate_hyperparameters(base, {}, {"first", "second"})
        self.assertEqual(settings.dataset_token_percentages["first"], 25.0)

        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            _validate_hyperparameters(
                {**base, "dataset_token_percentages": {"other": 100}},
                {},
                {"first", "second"},
            )
        with self.assertRaisesRegex(ValueError, "sum to 100"):
            _validate_hyperparameters(
                {**base, "dataset_token_percentages": {"first": 25, "second": 25}},
                {},
                {"first", "second"},
            )

    def test_budgets_sum_exactly_after_percentage_rounding(self):
        budgets = allocate_dataset_token_budgets(
            7,
            {"first": 33.3, "second": 66.7},
        )
        self.assertEqual(budgets, {"first": 2, "second": 5})
        self.assertEqual(sum(budgets.values()), 7)

    def test_prefix_tokenization_and_packing_use_exact_budget(self):
        source = Dataset.from_dict(
            {
                "text": ["abc", "defgh", "ijklmno"],
                DATASET_NAME_COLUMN: ["first", "first", "first"],
            }
        )
        tokenized = pretokenize_split(
            source,
            _LengthTokenizer(),
            tokenization_batch_size=2,
            max_tokens=5,
        )

        self.assertEqual(tokenized["token_ids"], [[1, 2, 3], [1, 2, 3]])
        self.assertTrue(tokenized.cache_files)
        self.assertEqual(tokenized[DATASET_NAME_COLUMN], ["first", "first"])
        batches = list(
            iter_dataset_packed_batches(
                {"first": tokenized},
                {"first": 5},
                sequence_length=4,
                packing_stride=4,
                pad_token_id=0,
                batch_size=1,
            )
        )
        self.assertEqual(sum(int(batch.target_mask.sum()) for _, batch in batches), 5)

    def test_unbudgeted_tokenization_keeps_empty_rows_on_disk(self):
        source = Dataset.from_dict(
            {
                "text": ["", "abc", None, "de"],
                DATASET_NAME_COLUMN: ["first"] * 4,
            }
        )

        tokenized = pretokenize_split(
            source,
            _LengthTokenizer(),
            tokenization_batch_size=2,
        )

        self.assertEqual(tokenized["token_ids"], [[], [1, 2, 3], [], [1, 2]])
        self.assertEqual(tokenized[DATASET_NAME_COLUMN], ["first"] * 4)
        self.assertTrue(tokenized.cache_files)


if __name__ == "__main__":
    unittest.main()

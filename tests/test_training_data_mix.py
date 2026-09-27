import sys
from pathlib import Path
import unittest
from unittest.mock import patch

from datasets import Dataset


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from datasets_mgr import (  # noqa: E402
    DATASET_NAME_COLUMN,
    get_dataset_sources,
    unified_dataset_loader,
)
import datasets_mgr  # noqa: E402
from train.hyperparameters import (  # noqa: E402
    _validate_hyperparameters,
)
from train.packing import (  # noqa: E402
    iter_dataset_packed_batches,
    packed_target_capacity,
    pretokenize_split,
)
from train.budgets import _plan_training_budgets, _planned_training_steps  # noqa: E402


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _LengthTokenizer:
    def encode_batch(self, texts, *, add_special_tokens):
        del add_special_tokens
        return [_Encoding(list(range(1, len(text) + 1))) for text in texts]


class TrainingDataMixTests(unittest.TestCase):
    def test_interleaved_source_names_preserve_row_order_with_filter_fallback(self):
        source = Dataset.from_dict(
            {
                "text": ["a1", "b1", "a2", "b2", "a3"],
                DATASET_NAME_COLUMN: ["a", "b", "a", "b", "a"],
            }
        )
        with patch.object(datasets_mgr, "_MAX_SOURCE_NAME_RANGES", 1):
            sources = get_dataset_sources(source)
        self.assertEqual(sources["a"]["text"], ["a1", "a2", "a3"])
        self.assertEqual(sources["b"]["text"], ["b1", "b2"])

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

    def test_count_validation_checks_names_values_and_sum(self):
        base = {
            "batch_size": 1,
            "learning_rate": 1e-3,
            "max_train_tokens": 10,
            "num_epochs": 1,
            "dataset_token_counts": {"first": 2, "second": 8},
        }
        settings = _validate_hyperparameters(base, {}, {"first", "second"})
        self.assertEqual(settings.dataset_token_counts["first"], 2)

        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            _validate_hyperparameters(
                {**base, "dataset_token_counts": {"other": 10}},
                {},
                {"first", "second"},
            )
        with self.assertRaisesRegex(ValueError, "missing dataset"):
            _validate_hyperparameters(
                {**base, "dataset_token_counts": {"first": 10}},
                {},
                {"first", "second"},
            )
        with self.assertRaisesRegex(ValueError, "sum to max_train_tokens"):
            _validate_hyperparameters(
                {**base, "dataset_token_counts": {"first": 2, "second": 7}},
                {},
                {"first", "second"},
            )
        for invalid in (True, 1.5, -1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "non-negative integers"
            ):
                _validate_hyperparameters(
                    {**base, "dataset_token_counts": {"first": invalid, "second": 8}},
                    {},
                    {"first", "second"},
                )

    def test_minibatch_and_accumulation_settings(self):
        base = {
            "batch_size": 8,
            "learning_rate": 1e-3,
            "max_train_tokens": 10,
            "num_epochs": 1,
            "dataset_token_counts": {"first": 10},
        }
        for extra in (
            {"minibatch_size": 2},
            {"gradient_accumulation_steps": 4},
            {"minibatch_size": 2, "gradient_accumulation_steps": 4},
        ):
            with self.subTest(extra=extra):
                settings = _validate_hyperparameters({**base, **extra}, {}, {"first"})
                self.assertEqual(settings.minibatch_size, 2)
                self.assertEqual(settings.gradient_accumulation_steps, 4)

        defaults = _validate_hyperparameters(base, {}, {"first"})
        self.assertEqual(defaults.minibatch_size, 8)
        self.assertEqual(defaults.gradient_accumulation_steps, 1)

        for extra in (
            {"minibatch_size": 0},
            {"minibatch_size": None},
            {"gradient_accumulation_steps": True},
            {"minibatch_size": 3},
            {"gradient_accumulation_steps": 3},
            {"minibatch_size": 2, "gradient_accumulation_steps": 3},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                _validate_hyperparameters({**base, **extra}, {}, {"first"})

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
        self.assertEqual(packed_target_capacity(tokenized, 4, 4), 5)

    def test_capacity_counts_overlapping_windows_and_short_streams(self):
        split = Dataset.from_dict({"token_ids": [[1, 2, 3], [4, 5, 6]]})
        self.assertEqual(packed_target_capacity(split, 4, 4), 5)
        self.assertEqual(packed_target_capacity(split, 4, 2), 7)
        self.assertEqual(
            packed_target_capacity(Dataset.from_dict({"token_ids": [[1]]}), 4, 4),
            0,
        )

    def test_absolute_counts_are_planned_across_epochs(self):
        base = {
            "batch_size": 1,
            "learning_rate": 1e-3,
            "max_train_tokens": 6,
            "num_epochs": 3,
            "sequence_length": 4,
            "dataset_token_counts": {"first": 5, "second": 1},
        }
        splits = {
            "first": Dataset.from_dict({"token_ids": [[1, 2, 3]]}),
            "second": Dataset.from_dict({"token_ids": [[1, 2, 3]]}),
        }
        settings = _validate_hyperparameters(base, {}, splits.keys())
        self.assertEqual(
            _plan_training_budgets(splits, settings),
            [
                {"first": 2, "second": 1},
                {"first": 2, "second": 0},
                {"first": 1, "second": 0},
            ],
        )
        emitted = 0
        for epoch_budgets in _plan_training_budgets(splits, settings):
            for _, batch in iter_dataset_packed_batches(
                splits,
                epoch_budgets,
                sequence_length=4,
                packing_stride=4,
                pad_token_id=0,
                batch_size=1,
            ):
                emitted += int(batch.target_mask.sum())
        self.assertEqual(emitted, settings.max_train_tokens)

        settings = _validate_hyperparameters(
            {**base, "num_epochs": 2}, {}, splits.keys()
        )
        with self.assertRaisesRegex(ValueError, "'first' is assigned 5"):
            _plan_training_budgets(splits, settings)

    def test_scheduler_counts_partial_batches_in_each_epoch(self):
        budgets = [{"first": 1}, {"first": 1}, {"first": 1}, {"first": 1}]
        self.assertEqual(
            _planned_training_steps(budgets, sequence_length=4, batch_size=8),
            4,
        )

    def test_unloaded_zero_budget_source_needs_no_tokenized_split(self):
        settings = _validate_hyperparameters(
            {
                "batch_size": 1,
                "learning_rate": 1e-3,
                "max_train_tokens": 2,
                "num_epochs": 1,
                "sequence_length": 4,
                "dataset_token_counts": {"first": 2, "unused": 0},
            },
            {},
            {"first", "unused"},
        )
        splits = {"first": Dataset.from_dict({"token_ids": [[1, 2, 3]]})}
        budgets = _plan_training_budgets(splits, settings)[0]

        batches = list(
            iter_dataset_packed_batches(
                splits,
                budgets,
                sequence_length=4,
                packing_stride=4,
                pad_token_id=0,
                batch_size=1,
            )
        )
        self.assertEqual(sum(int(batch.target_mask.sum()) for _, batch in batches), 2)

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

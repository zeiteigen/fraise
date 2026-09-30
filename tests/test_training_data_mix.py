import unittest

from datasets import Dataset
from fixtures import tiny_settings, tiny_tokenizer

from fraise.training.batches import iter_dataset_packed_batches
from fraise.training.budgets import _plan_training_budgets, _planned_training_steps
from fraise.training.packing import iter_packed_blocks, packed_target_capacity
from fraise.training.tokenization import pretokenize_split


class TrainingDataMixTests(unittest.TestCase):
    def test_prefix_tokenization_and_exact_source_budget(self):
        source = Dataset.from_dict({"text": ["a b", "c d e", "a b c"]})
        tokenized = pretokenize_split(
            source, tiny_tokenizer(), tokenization_batch_size=2, max_tokens=5
        )
        self.assertEqual(tokenized["token_ids"], [[1, 3, 4, 2], [1, 5]])
        self.assertTrue(tokenized.cache_files)
        batches = list(
            iter_dataset_packed_batches(
                {"fixture": tokenized}, {"fixture": 5}, 4, 4, 0, batch_size=1
            )
        )
        self.assertEqual(sum(int(batch.target_mask.sum()) for _, batch in batches), 5)
        self.assertEqual(
            [batch.sources for _, batch in batches], [("fixture",), ("fixture",)]
        )

    def test_capacity_matches_emitted_windows_for_overlaps_and_tails(self):
        for length in range(1, 25):
            split = Dataset.from_dict({"token_ids": [list(range(length))]})
            for context in range(2, 7):
                for stride in range(1, context + 1):
                    with self.subTest(length=length, context=context, stride=stride):
                        blocks = iter_packed_blocks(
                            split,
                            context,
                            stride,
                            0,
                            seed=0,
                            shuffle=False,
                            max_tokens=None,
                        )
                        self.assertEqual(
                            packed_target_capacity(split, context, stride),
                            sum(int(block.target_mask.sum()) for block in blocks),
                        )

    def test_source_counts_repeat_in_full_each_epoch(self):
        settings = tiny_settings(
            dataset_token_counts={"first": 5, "second": 1},
            max_train_tokens=6,
            num_epochs=3,
            sequence_length=4,
        )
        splits = {
            "first": Dataset.from_dict({"token_ids": [[1, 2, 3, 4, 5, 6]]}),
            "second": Dataset.from_dict({"token_ids": [[1, 2, 3]]}),
        }
        budgets = _plan_training_budgets(splits, settings)
        self.assertEqual(
            budgets,
            [
                {"first": 5, "second": 1},
                {"first": 5, "second": 1},
                {"first": 5, "second": 1},
            ],
        )
        emitted = sum(
            int(batch.target_mask.sum())
            for budget in budgets
            for _, batch in iter_dataset_packed_batches(
                splits, budget, 4, 4, 0, batch_size=2
            )
        )
        self.assertEqual(emitted, 18)

    def test_more_epochs_do_not_relax_per_epoch_source_capacity(self):
        splits = {
            "first": Dataset.from_dict({"token_ids": [[1, 2, 3]]}),
            "second": Dataset.from_dict({"token_ids": [[1, 2, 3]]}),
        }
        for num_epochs in (1, 3, 10):
            with (
                self.subTest(num_epochs=num_epochs),
                self.assertRaisesRegex(ValueError, "'first' is assigned 5"),
            ):
                _plan_training_budgets(
                    splits,
                    tiny_settings(
                        dataset_token_counts={"first": 5, "second": 1},
                        max_train_tokens=6,
                        num_epochs=num_epochs,
                        sequence_length=4,
                    ),
                )

    def test_resume_cursor_preserves_batches_and_source_order(self):
        splits = {
            "first": Dataset.from_dict({"token_ids": [[1, 2, 3, 4, 5, 6]]}),
            "second": Dataset.from_dict({"token_ids": [[2, 3, 4, 5]]}),
        }
        kwargs = {"batch_size": 2, "seed": 3, "shuffle_per_epoch": True}
        all_batches = list(
            iter_dataset_packed_batches(
                splits, {"first": 5, "second": 3}, 2, 2, 0, **kwargs
            )
        )
        resumed = list(
            iter_dataset_packed_batches(
                splits,
                {"first": 5, "second": 3},
                2,
                2,
                0,
                start_batch_index=1,
                **kwargs,
            )
        )
        self.assertEqual(
            [index for index, _ in resumed], [index for index, _ in all_batches[1:]]
        )
        for (_, actual), (_, expected) in zip(resumed, all_batches[1:]):
            self.assertEqual(actual.sources, expected.sources)
            self.assertTrue(actual.input_ids.equal(expected.input_ids))

    def test_scheduler_counts_partial_source_and_epoch_batches(self):
        self.assertEqual(
            _planned_training_steps(
                [{"first": 1}] * 4, sequence_length=4, batch_size=8
            ),
            4,
        )
        self.assertEqual(
            _planned_training_steps(
                [{"first": 5, "second": 3}], sequence_length=4, batch_size=2
            ),
            2,
        )

    def test_zero_budget_sources_need_no_split(self):
        settings = tiny_settings(dataset_token_counts={"fixture": 4, "unused": 0})
        budgets = _plan_training_budgets(
            {"fixture": Dataset.from_dict({"token_ids": [[1, 2, 3, 4, 5]]})}, settings
        )
        self.assertEqual(budgets, [{"fixture": 4, "unused": 0}])

    def test_unbudgeted_tokenization_keeps_empty_rows(self):
        split = pretokenize_split(
            Dataset.from_dict({"text": ["", "a b", None, "c"]}),
            tiny_tokenizer(),
            tokenization_batch_size=2,
        )
        self.assertEqual(split["token_ids"], [[], [1, 3, 4, 2], [], [1, 5, 2]])
        self.assertTrue(split.cache_files)

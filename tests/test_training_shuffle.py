import unittest
from collections import Counter

from datasets import Dataset

from fraise.training.batches import iter_dataset_packed_batches
from fraise.training.mixing import iter_mixed_source_blocks


def window_rows(batches):
    return [
        (source, tuple(inputs.tolist()), tuple(targets.tolist()), tuple(mask.tolist()))
        for _, batch in batches
        for source, inputs, targets, mask in zip(
            batch.sources, batch.input_ids, batch.targets, batch.target_mask
        )
    ]


class TrainingShuffleTests(unittest.TestCase):
    def setUp(self):
        self.splits = {
            "first": Dataset.from_dict(
                {"token_ids": [list(range(10, 19)), list(range(20, 29))]}
            ),
            "second": Dataset.from_dict(
                {"token_ids": [list(range(30, 35)), list(range(40, 45))]}
            ),
            "third": Dataset.from_dict({"token_ids": [[50, 51, 52, 53]]}),
            "unused": Dataset.from_dict({"unused": []}),
        }
        self.budgets = {"first": 17, "second": 9, "third": 3, "unused": 0}

    def batches(self, **options):
        return iter_dataset_packed_batches(
            self.splits, self.budgets, 4, 4, 0, batch_size=2, seed=23, **options
        )

    def test_global_mixing_preserves_each_sources_windows_and_exact_target_counts(self):
        for per_epoch in (False, True):
            with self.subTest(per_epoch=per_epoch):
                ordered = window_rows(self.batches(shuffle_per_epoch=per_epoch))
                mixed_batches = list(
                    self.batches(shuffle_per_epoch=per_epoch, shuffle_data=True)
                )
                mixed = window_rows(mixed_batches)
                self.assertEqual(Counter(mixed), Counter(ordered))
                self.assertNotEqual(mixed, ordered)
                self.assertTrue(
                    any(len(set(batch.sources)) > 1 for _, batch in mixed_batches)
                )
                for name, budget in self.budgets.items():
                    self.assertEqual(
                        [row for row in mixed if row[0] == name],
                        [row for row in ordered if row[0] == name],
                    )
                    self.assertEqual(
                        sum(sum(row[3]) for row in mixed if row[0] == name), budget
                    )

    def test_per_epoch_shuffle_reorders_passages_before_packing(self):
        split = {
            "first": Dataset.from_dict(
                {"token_ids": [[i, i + 1, i + 2] for i in range(0, 24, 3)]}
            )
        }
        options = {"batch_size": 1, "seed": 2}
        ordered = window_rows(
            iter_dataset_packed_batches(split, {"first": 23}, 4, 4, 0, **options)
        )
        shuffled = window_rows(
            iter_dataset_packed_batches(
                split, {"first": 23}, 4, 4, 0, shuffle_per_epoch=True, **options
            )
        )
        self.assertNotEqual(ordered, shuffled)

    def test_mixed_order_and_resume_cursor_are_reproducible(self):
        for per_epoch in (False, True):
            with self.subTest(per_epoch=per_epoch):
                options = {"shuffle_per_epoch": per_epoch, "shuffle_data": True}
                original = list(self.batches(**options))
                self.assertEqual(
                    window_rows(original), window_rows(self.batches(**options))
                )
                for start in (1, 3, 4):
                    resumed = list(self.batches(start_batch_index=start, **options))
                    self.assertEqual(
                        [index for index, _ in resumed],
                        [index for index, _ in original[start:]],
                    )
                    self.assertEqual(
                        window_rows(resumed), window_rows(original[start:])
                    )

    def test_mixer_does_not_materialize_the_entire_corpus(self):
        consumed = Counter()

        def blocks(name):
            for index in range(1000000):
                consumed[name] += 1
                yield index

        mixed = iter_mixed_source_blocks(
            {name: blocks(name) for name in ("first", "second")},
            {"first": 1000000, "second": 1000000},
            seed=23,
        )
        next(mixed)
        self.assertEqual(sum(consumed.values()), 1)

    def test_mixer_rejects_a_source_that_cannot_supply_its_allocation(self):
        mixed = iter_mixed_source_blocks({"first": iter([1])}, {"first": 2}, seed=23)
        self.assertEqual(next(mixed), ("first", 1))
        with self.assertRaisesRegex(RuntimeError, "'first' produced fewer windows"):
            next(mixed)


if __name__ == "__main__":
    unittest.main()

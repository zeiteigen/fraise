import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from datasets import Dataset
from fixtures import tiny_settings
from macro_fixtures import tiny_macro_tokenizer

from fraise.training.pipeline.batch_types import target_units
from fraise.training.pipeline.batches import iter_dataset_packed_batches
from fraise.training.pipeline.budgets import plan_epochs
from fraise.training.pipeline.packing import iter_packed_blocks, plan_unit_windows
from fraise.training.pipeline.tokenization import pretokenize_split


def reference_windows(stream, units, length, stride, budget):
    """Window-by-window unit accounting written for clarity, not speed."""
    n, produced, windows = len(stream), 0, []
    starts = list(range(0, n - length, stride)) if n >= length + 1 else []
    full = [s for s in starts if s + length + 1 <= n]
    tail = full[-1] + stride if full else 0
    spans = [(s, length) for s in full]
    if n - tail >= 2 and (not full or tail > full[-1]):
        spans.append((tail, n - tail - 1))
    for start, available in spans:
        targets = stream[start + 1 : start + 1 + available]
        count = 0
        for token in targets:
            if produced >= budget:
                break
            produced += units[token]
            count += 1
        windows.append((start, count))
        if produced >= budget:
            break
    return windows, produced


class UnitPackingTests(unittest.TestCase):
    def test_unit_windows_match_reference_and_resume(self):
        rng = np.random.default_rng(3)
        units = np.array([1, 1, 1, 2, 3, 1, 4, 2], dtype=np.int64)
        documents = [rng.integers(0, 8, rng.integers(1, 9)).tolist() for _ in range(9)]
        split = Dataset.from_dict({"token_ids": documents})
        stream = [token for document in documents for token in document]
        capacity = reference_windows(stream, units, 4, 4, float("inf"))[1]
        for length, stride in ((4, 4), (4, 2), (3, 1)):
            for budget in (1, 5, 11, 23, capacity, None):
                with self.subTest(length=length, stride=stride, budget=budget):
                    limit = float("inf") if budget is None else budget
                    expected, units_total = reference_windows(
                        stream, units, length, stride, limit
                    )
                    plan = plan_unit_windows(split, length, stride, budget, units)
                    self.assertEqual(plan.windows, len(expected))
                    self.assertEqual(plan.units, units_total)
                    options = {"seed": 0, "shuffle": False, "max_tokens": budget}
                    blocks = list(
                        iter_packed_blocks(
                            split, length, stride, 0, unit_lengths=units, **options
                        )
                    )
                    self.assertEqual(len(blocks), len(expected))
                    for block, (start, count) in zip(blocks, expected):
                        self.assertEqual(int(block.target_mask.sum()), count)
                        self.assertEqual(
                            block.targets[:count].tolist(),
                            stream[start + 1 : start + 1 + count],
                        )
                        self.assertEqual(block.input_ids[0].item(), stream[start])
                    self.assertEqual(
                        sum(target_units(block, units) for block in blocks),
                        units_total,
                    )
                    for resume in range(len(blocks) + 1):
                        resumed = list(
                            iter_packed_blocks(
                                split,
                                length,
                                stride,
                                0,
                                unit_lengths=units,
                                start_window=resume,
                                **options,
                            )
                        )
                        self.assertEqual(
                            [b.targets.tolist() for b in resumed],
                            [b.targets.tolist() for b in blocks[resume:]],
                        )

    def test_epoch_plans_use_exact_units_and_reject_short_sources(self):
        units = np.array([1, 1, 1, 3, 2], dtype=np.int64)
        split = Dataset.from_dict({"token_ids": [[1, 3, 4, 3, 2], [1, 4, 4, 3, 2]]})
        settings = tiny_settings(
            dataset_token_counts={"fixture": 7},
            max_train_tokens=7,
            sequence_length=3,
            num_epochs=2,
            batch_size=2,
        )
        budgets, windows = plan_epochs({"fixture": split}, settings, unit_lengths=units)
        # Targets 3,4,3 cover 3+2+3=8 units: the macro that reaches 7 is kept.
        self.assertEqual(budgets, [{"fixture": 8}] * 2)
        self.assertEqual(windows, [{"fixture": 1}] * 2)
        batches = list(
            iter_dataset_packed_batches(
                {"fixture": split},
                budgets[0],
                3,
                3,
                0,
                batch_size=2,
                unit_lengths=units,
                window_counts=windows[0],
            )
        )
        self.assertEqual(sum(target_units(b, units) for _, b in batches), 8)
        too_large = tiny_settings(
            dataset_token_counts={"fixture": 99}, max_train_tokens=99
        )
        with self.assertRaisesRegex(ValueError, "can supply at most"):
            plan_epochs({"fixture": split}, too_large, unit_lengths=units)

    def test_pretokenization_keeps_the_macro_crossing_the_unit_budget(self):
        macro = tiny_macro_tokenizer()
        text = "a b c d e a b c d e a b c d e"
        with (
            TemporaryDirectory() as cache,
            patch("datasets.config.HF_DATASETS_CACHE", Path(cache)),
        ):
            for budget in (1, 4, 9):
                with self.subTest(budget=budget):
                    split = pretokenize_split(
                        Dataset.from_dict({"text": [text, text]}),
                        macro,
                        tokenization_batch_size=1,
                        max_tokens=budget,
                    )
                    ids = split["token_ids"][0]
                    covered = int(macro.unit_lengths[ids].sum())
                    self.assertGreaterEqual(covered, budget + 1)
                    self.assertLess(int(macro.unit_lengths[ids[:-1]].sum()), budget + 1)
                    self.assertEqual(
                        macro.expand(ids), macro.base.encode(text).ids[:covered]
                    )


if __name__ == "__main__":
    unittest.main()

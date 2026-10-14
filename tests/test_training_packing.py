"""Vectorized packing matches the original streaming packer exactly."""

import random
import unittest
from collections import deque
from itertools import islice

import torch
from datasets import Dataset

from fraise.training.pipeline.batches import iter_dataset_packed_batches
from fraise.training.pipeline.packing import iter_packed_blocks


def reference_blocks(documents, sequence_length, stride, pad, max_tokens):
    """The pre-vectorization deque packer, kept as the behavioral oracle."""

    def block(tokens, valid):
        tokens = tokens[: valid + 1] + [pad] * (sequence_length + 1 - (valid + 1))
        mask = [True] * valid + [False] * (sequence_length - valid)
        return (tokens[:-1], tokens[1:], mask)

    buffer, produced = deque(), 0
    for document in documents:
        if not document:
            continue
        buffer.extend(document)
        while len(buffer) >= sequence_length + 1:
            remaining = (
                max_tokens - produced if max_tokens is not None else sequence_length
            )
            if remaining <= 0:
                return
            valid = min(sequence_length, remaining)
            yield block(list(islice(buffer, valid + 1)), valid)
            produced += valid
            if valid < sequence_length:
                return
            for _ in range(stride):
                buffer.popleft()
    if max_tokens is not None and produced >= max_tokens:
        return
    if len(buffer) < 2:
        return
    remaining = max_tokens - produced if max_tokens is not None else sequence_length
    valid = min(sequence_length, len(buffer) - 1, remaining)
    if valid > 0:
        yield block(list(islice(buffer, valid + 1)), valid)


def rows(blocks):
    return [
        (b.input_ids.tolist(), b.targets.tolist(), b.target_mask.tolist())
        for b in blocks
    ]


class PackingEquivalenceTests(unittest.TestCase):
    def test_matches_reference_for_geometries_budgets_shuffles_and_resumes(self):
        generator = random.Random(11)
        for trial in range(40):
            documents = [
                [generator.randrange(3, 50) for _ in range(generator.randrange(0, 9))]
                for _ in range(generator.randrange(1, 12))
            ]
            split = Dataset.from_dict({"token_ids": documents})
            total = sum(map(len, documents))
            context = generator.randrange(2, 7)
            stride = generator.randrange(1, context + 1)
            budget = generator.choice([None, generator.randrange(1, total + 2)])
            shuffle = generator.random() < 0.5
            ordered = split.shuffle(seed=trial) if shuffle else split
            expected = list(
                reference_blocks(ordered["token_ids"], context, stride, 0, budget)
            )
            options = dict(seed=trial, shuffle=shuffle, max_tokens=budget)
            with self.subTest(trial=trial, context=context, stride=stride):
                actual = rows(iter_packed_blocks(split, context, stride, 0, **options))
                self.assertEqual(actual, expected)
                for start in range(len(expected) + 2):
                    resumed = iter_packed_blocks(
                        split, context, stride, 0, start_window=start, **options
                    )
                    self.assertEqual(rows(resumed), expected[start:])

    def test_int32_and_selected_splits_produce_int64_windows(self):
        from datasets import Features, List, Value

        split = Dataset.from_dict(
            {"token_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]},
            features=Features({"token_ids": List(Value("int32"))}),
        ).select([2, 0])
        blocks = list(
            iter_packed_blocks(split, 3, 3, 0, seed=0, shuffle=False, max_tokens=None)
        )
        self.assertEqual(blocks[0].input_ids.dtype, torch.long)
        self.assertEqual(
            rows(blocks),
            list(reference_blocks([[6, 7, 8, 9], [1, 2, 3]], 3, 3, 0, None)),
        )

    def test_every_resume_index_matches_the_uninterrupted_batches(self):
        splits = {
            "first": Dataset.from_dict(
                {"token_ids": [list(range(i, i + 7)) for i in range(0, 70, 7)]}
            ),
            "second": Dataset.from_dict({"token_ids": [list(range(100, 131))]}),
        }
        budgets = {"first": 41, "second": 17}
        for stride in (2, 4):
            for shuffle_per_epoch in (False, True):
                for shuffle_data in (False, True):
                    options = dict(
                        batch_size=3,
                        seed=5,
                        shuffle_per_epoch=shuffle_per_epoch,
                        shuffle_data=shuffle_data,
                    )
                    full = list(
                        iter_dataset_packed_batches(
                            splits, budgets, 4, stride, 0, **options
                        )
                    )
                    for start in range(len(full) + 1):
                        with self.subTest(
                            stride=stride,
                            per_epoch=shuffle_per_epoch,
                            mixed=shuffle_data,
                            start=start,
                        ):
                            resumed = list(
                                iter_dataset_packed_batches(
                                    splits,
                                    budgets,
                                    4,
                                    stride,
                                    0,
                                    start_batch_index=start,
                                    **options,
                                )
                            )
                            self.assertEqual(
                                [index for index, _ in resumed],
                                list(range(start, len(full))),
                            )
                            for (_, actual), (_, expected) in zip(
                                resumed, full[start:]
                            ):
                                self.assertEqual(actual.sources, expected.sources)
                                torch.testing.assert_close(
                                    actual.input_ids, expected.input_ids
                                )
                                torch.testing.assert_close(
                                    actual.targets, expected.targets
                                )
                                torch.testing.assert_close(
                                    actual.target_mask, expected.target_mask
                                )


if __name__ == "__main__":
    unittest.main()

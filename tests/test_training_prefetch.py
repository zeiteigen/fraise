import sys
from pathlib import Path
from threading import Event
import unittest

import torch


SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from train.packing import PackedTokenBatch  # noqa: E402
from train.prefetch import prefetch_batches  # noqa: E402


class TrainingPrefetchTests(unittest.TestCase):
    def test_next_batch_is_prepared_while_consumer_uses_current_batch(self):
        preparing_second = Event()
        batch = PackedTokenBatch(
            input_ids=torch.tensor([[1, 2]]),
            targets=torch.tensor([[2, 3]]),
            target_mask=torch.tensor([[True, True]]),
        )

        def source():
            yield 0, batch
            preparing_second.set()
            yield 1, batch

        prefetched = prefetch_batches(source(), pin_memory=False)
        first_index, first_batch = next(prefetched)
        self.assertEqual(first_index, 0)
        self.assertIs(first_batch, batch)
        self.assertTrue(preparing_second.wait(timeout=2))
        self.assertEqual(next(prefetched)[0], 1)
        with self.assertRaises(StopIteration):
            next(prefetched)


if __name__ == "__main__":
    unittest.main()

"""Group named source windows into batches with a resumable cursor."""

from collections import Counter
from itertools import islice

import torch

from .batch_types import PackedTokenBatch
from .mixing import iter_source_blocks, mixed_source_order, sequential_source_order
from .packing import iter_packed_blocks


def iter_dataset_packed_batches(
    splits,
    token_budgets,
    sequence_length,
    packing_stride,
    pad_token_id,
    *,
    batch_size,
    start_batch_index=0,
    seed=0,
    shuffle_per_epoch=False,
    shuffle_data=False,
):
    # Each source yields ceil(budget / sequence_length) windows: all are full
    # except possibly its last.
    counts = {
        name: (token_budgets[name] + sequence_length - 1) // sequence_length
        for name in splits
        if token_budgets[name] > 0
    }
    order = (
        mixed_source_order(counts, seed=seed)
        if shuffle_data
        else sequential_source_order(counts)
    )
    # Resuming replays only source labels, then opens each source at its window.
    skipped = Counter(islice(order, start_batch_index * batch_size))
    sources = {
        name: iter_packed_blocks(
            splits[name],
            sequence_length,
            packing_stride,
            pad_token_id,
            seed=seed,
            shuffle=shuffle_per_epoch,
            max_tokens=token_budgets[name],
            start_window=skipped[name],
        )
        for name in counts
    }
    windows = iter_source_blocks(sources, order)
    batch_index = start_batch_index
    while blocks := list(islice(windows, batch_size)):
        yield (
            batch_index,
            PackedTokenBatch(
                torch.stack([block.input_ids for _, block in blocks]),
                torch.stack([block.targets for _, block in blocks]),
                torch.stack([block.target_mask for _, block in blocks]),
                tuple(name for name, _ in blocks),
            ),
        )
        batch_index += 1

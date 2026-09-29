"""Group named source windows into batches with a resumable cursor."""

from itertools import islice

import torch

from .batch_types import PackedTokenBatch
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
    shuffle=False,
):
    def windows():
        for name, split in splits.items():
            budget = token_budgets[name]
            if budget == 0:
                continue
            for block in iter_packed_blocks(
                split,
                sequence_length,
                packing_stride,
                pad_token_id,
                seed=seed,
                shuffle=shuffle,
                max_tokens=budget,
            ):
                yield name, block

    source = windows()
    batch_index = 0
    while blocks := list(islice(source, batch_size)):
        if batch_index >= start_batch_index:
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

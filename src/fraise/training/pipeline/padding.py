"""Keep compiled minibatches at one shape without adding training targets."""

import torch

from .batch_types import PackedTokenBatch


def pad_minibatch(batch: PackedTokenBatch, size: int) -> PackedTokenBatch:
    missing = size - len(batch.input_ids)
    if missing <= 0:
        return batch
    shape = (missing, batch.input_ids.size(1))
    padded = PackedTokenBatch(
        torch.cat((batch.input_ids, batch.input_ids.new_zeros(shape))),
        torch.cat((batch.targets, batch.targets.new_zeros(shape))),
        torch.cat((batch.target_mask, batch.target_mask.new_zeros(shape))),
        batch.sources + ("",) * missing if batch.sources else (),
    )
    return padded.pin_memory() if batch.input_ids.is_pinned() else padded

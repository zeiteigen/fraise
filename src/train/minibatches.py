"""Accumulate token-weighted gradients over bounded minibatches."""

import torch

from components.model import FraiseModel

from .batch import _autocast_context, _loss_for_batch, _move_batch
from .packing import PackedTokenBatch


def _backward_minibatches(
    model: FraiseModel,
    cpu_batch: PackedTokenBatch,
    model_device: torch.device,
    *,
    minibatch_size: int,
    use_bf16: bool,
) -> tuple[torch.Tensor, int] | None:
    """Accumulate the full batch's token-mean gradient one minibatch at a time."""
    total_valid_tokens = int(cpu_batch.target_mask.sum().item())
    if total_valid_tokens == 0:
        return None

    loss_sum: torch.Tensor | None = None
    for start in range(0, len(cpu_batch.input_ids), minibatch_size):
        stop = start + minibatch_size
        minibatch = PackedTokenBatch(
            input_ids=cpu_batch.input_ids[start:stop],
            targets=cpu_batch.targets[start:stop],
            target_mask=cpu_batch.target_mask[start:stop],
        )
        minibatch_valid_tokens = int(minibatch.target_mask.sum().item())
        if minibatch_valid_tokens == 0:
            continue
        batch = _move_batch(minibatch, model_device)
        with _autocast_context(use_bf16):
            result = _loss_for_batch(
                model, batch, valid_count=minibatch_valid_tokens
            )
        if result is None:
            continue

        loss, valid_tokens = result
        (loss * (valid_tokens / total_valid_tokens)).backward()
        contribution = loss.detach() * valid_tokens
        loss_sum = contribution if loss_sum is None else loss_sum + contribution

    assert loss_sum is not None
    return loss_sum / total_valid_tokens, total_valid_tokens

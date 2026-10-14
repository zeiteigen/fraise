"""Accumulate token-weighted gradients over bounded minibatches."""

from time import monotonic

import torch

from .batch import autocast_context, loss_for_batch
from .pipeline.padding import pad_minibatch


def backward_minibatches(
    model,
    cpu_batch,
    model_device,
    *,
    minibatch_size,
    use_bf16,
    pad_to_minibatch_size=False,
    on_first_backward=None,
    loss_sum=None,
):
    total_tokens = int(cpu_batch.target_mask.sum().item())
    if total_tokens <= 0:
        raise ValueError("packed batches must contain valid targets")
    total_loss = None
    for start in range(0, len(cpu_batch.input_ids), minibatch_size):
        cpu_minibatch = cpu_batch.slice(start, start + minibatch_size)
        valid_tokens = int(cpu_minibatch.target_mask.sum().item())
        if valid_tokens == 0:
            continue
        if pad_to_minibatch_size:
            cpu_minibatch = pad_minibatch(cpu_minibatch, minibatch_size)
        batch = cpu_minibatch.to(model_device)
        started = monotonic() if on_first_backward is not None else None
        with autocast_context(model_device, use_bf16):
            loss = loss_for_batch(model, batch, valid_tokens, loss_sum=loss_sum)
        (loss * (valid_tokens / total_tokens)).backward()
        if on_first_backward is not None:
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            on_first_backward(monotonic() - started)
            on_first_backward = None
        contribution = loss.detach() * valid_tokens
        total_loss = contribution if total_loss is None else total_loss + contribution
    return total_loss / total_tokens, total_tokens

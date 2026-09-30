"""Evaluate token-weighted held-out loss with bounded minibatches."""

from time import monotonic

import torch

from .batch import autocast_context, loss_for_batch
from .batches import iter_dataset_packed_batches
from .packing import packed_target_capacity


def evaluate(model, split, settings, pad_token_id, *, progress=None):
    if settings.max_validation_tokens == 0:
        return None
    device = next(model.parameters()).device
    capacity = packed_target_capacity(
        split, settings.sequence_length, settings.packing_stride
    )
    budget = (
        min(capacity, settings.max_validation_tokens)
        if settings.max_validation_tokens is not None
        else capacity
    )
    total_loss = total_tokens = 0
    was_training = model.training
    started = last_report = monotonic()
    if progress is not None:
        progress(f"Evaluating {budget:,} validation targets.")
    model.eval()
    try:
        with torch.inference_mode():
            for _, batch in iter_dataset_packed_batches(
                {"validation": split},
                {"validation": budget},
                settings.sequence_length,
                settings.packing_stride,
                pad_token_id,
                batch_size=settings.minibatch_size,
            ):
                valid_tokens = int(batch.target_mask.sum().item())
                with autocast_context(device, settings.use_bf16):
                    loss = loss_for_batch(model, batch.to(device), valid_tokens)
                total_loss += float(loss.item()) * valid_tokens
                total_tokens += valid_tokens
                now = monotonic()
                if progress is not None and (
                    now - last_report >= 10 or total_tokens == budget
                ):
                    elapsed = now - started
                    progress(
                        f"Progress {total_tokens:,}/{budget:,} targets, "
                        f"{elapsed:.1f}s elapsed, {total_tokens / max(elapsed, 0.001):,.0f} targets/s."
                    )
                    last_report = now
    finally:
        model.train(was_training)
    return total_loss / total_tokens if total_tokens else None

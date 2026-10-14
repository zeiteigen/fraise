"""Token-summed causal loss and autocast for the actual batch device."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def token_loss_sum(model, input_ids, targets, target_mask):
    """Sum cross-entropy over valid targets with shapes that never depend on data.

    Masked targets are ignored rather than gathered out, so there is no host sync
    and a compiled graph sees the same shapes for full, short, and padded inputs.
    """
    logits = model(input_ids)
    targets = targets.masked_fill(~target_mask, IGNORE_INDEX)
    return F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def compile_loss_sum():
    """Compile the forward pass and loss together as one fixed-shape graph."""
    return torch.compile(token_loss_sum, dynamic=False)


def loss_for_batch(model, batch, valid_count: int | None = None, *, loss_sum=None):
    if valid_count is None:
        valid_count = int(batch.target_mask.sum().item())
    if valid_count <= 0:
        raise ValueError("packed batches must contain valid targets")
    loss_sum = loss_sum or token_loss_sum
    # Divide outside the compiled graph so the token count is never a guard.
    loss = loss_sum(model, batch.input_ids, batch.targets, batch.target_mask)
    loss = loss / valid_count
    if loss.device.type == "cpu" and not torch.isfinite(loss).item():
        raise FloatingPointError("non-finite loss detected")
    return loss


def autocast_context(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if enabled
        else nullcontext()
    )

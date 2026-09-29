"""Token-mean causal loss and autocast for the actual batch device."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def loss_for_batch(model, batch, valid_count: int | None = None):
    if valid_count is None:
        valid_count = int(batch.target_mask.sum().item())
    if valid_count <= 0:
        raise ValueError("packed batches must contain valid targets")
    logits = model(batch.input_ids).flatten(0, 1)
    targets = batch.targets.flatten()
    if valid_count != targets.numel():
        mask = batch.target_mask.flatten()
        logits, targets = logits[mask], targets[mask]
    loss = F.cross_entropy(logits, targets)
    if loss.device.type == "cpu" and not torch.isfinite(loss).item():
        raise FloatingPointError("non-finite loss detected")
    return loss


def autocast_context(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if enabled
        else nullcontext()
    )

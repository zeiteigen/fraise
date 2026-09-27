"""Operations shared by training and validation batches."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from components import device
from components.model import FraiseModel

from .packing import PackedTokenBatch


def _loss_for_batch(
    model: FraiseModel,
    batch: PackedTokenBatch,
    *,
    valid_count: int | None = None,
) -> tuple[torch.Tensor, int] | None:
    """Return the mean cross-entropy over non-padding targets."""
    if valid_count is None:
        valid_count = int(batch.target_mask.sum().item())
    if valid_count == 0:
        return None

    logits = model(batch.input_ids)
    flat_targets = batch.targets.reshape(-1)
    flat_logits = logits.reshape(-1, logits.size(-1))
    if valid_count == flat_targets.numel():
        loss = F.cross_entropy(flat_logits, flat_targets)
    else:
        valid_targets = batch.target_mask.reshape(-1)
        loss = F.cross_entropy(
            flat_logits[valid_targets], flat_targets[valid_targets]
        )
    # Training checks gradients after backward. CPU validation can still give
    # a direct loss error without synchronizing a CUDA stream.
    if loss.device.type == "cpu" and not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("non-finite loss detected")
    return loss, valid_count


def _autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _move_batch(
    batch: PackedTokenBatch,
    target_device: torch.device,
) -> PackedTokenBatch:
    return PackedTokenBatch(
        input_ids=batch.input_ids.to(target_device, non_blocking=True),
        targets=batch.targets.to(target_device, non_blocking=True),
        target_mask=batch.target_mask.to(target_device, non_blocking=True),
    )

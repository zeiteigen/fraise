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
) -> tuple[torch.Tensor, int] | None:
    """Return the mean cross-entropy over non-padding targets."""
    logits = model(batch.input_ids)

    valid_targets = batch.target_mask.reshape(-1)
    valid_count = int(valid_targets.sum().item())
    if valid_count == 0:
        return None

    flat_targets = batch.targets.reshape(-1)
    selected_targets = flat_targets[valid_targets]
    if int(selected_targets.min().item()) < 0 or int(
        selected_targets.max().item()
    ) >= logits.size(-1):
        raise ValueError("target token ID is outside the model vocabulary")

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1))[valid_targets],
        selected_targets,
    )
    if not bool(torch.isfinite(loss).item()):
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

"""Packed tensors and their source labels."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PackedTokenBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    sources: tuple[str, ...] = ()

    def slice(self, start: int, stop: int) -> "PackedTokenBatch":
        return PackedTokenBatch(
            self.input_ids[start:stop],
            self.targets[start:stop],
            self.target_mask[start:stop],
            self.sources[start:stop],
        )

    def to(self, device: torch.device) -> "PackedTokenBatch":
        return PackedTokenBatch(
            self.input_ids.to(device, non_blocking=True),
            self.targets.to(device, non_blocking=True),
            self.target_mask.to(device, non_blocking=True),
            self.sources,
        )

    def pin_memory(self) -> "PackedTokenBatch":
        return PackedTokenBatch(
            self.input_ids.pin_memory(),
            self.targets.pin_memory(),
            self.target_mask.pin_memory(),
            self.sources,
        )


def target_units(batch: PackedTokenBatch, unit_lengths: np.ndarray | None) -> int:
    """Budget units of a CPU batch: base tokens covered by its valid targets."""
    if unit_lengths is None:
        return int(batch.target_mask.sum().item())
    targets = batch.targets[batch.target_mask].numpy()
    return int(unit_lengths[targets].sum())

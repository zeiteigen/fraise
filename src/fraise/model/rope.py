"""Rotary position tables and query/key rotation."""

import math

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        self.dim = dim

        # Keep integer indices so model.to(dtype=...) does not reduce their
        # precision.
        self.register_buffer(
            "frequency_indices",
            torch.arange(0, dim, 2, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=compute_dtype,
        )
        indices = self.frequency_indices.to(
            device=device,
            dtype=compute_dtype,
        )

        inv_freq = torch.exp(-indices * (math.log(10_000.0) / self.dim))
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]

        return (
            cos,
            sin,
        )

    def apply_rotary_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        return (
            self._apply_rotary(q, cos, sin),
            self._apply_rotary(k, cos, sin),
        )

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = x.dtype

        x = x.to(cos.dtype).reshape(
            *x.shape[:-1],
            self.dim // 2,
            2,
        )
        even = x[..., 0]
        odd = x[..., 1]

        x = torch.stack(
            (
                even * cos - odd * sin,
                even * sin + odd * cos,
            ),
            dim=-1,
        )
        return x.flatten(-2).to(original_dtype)

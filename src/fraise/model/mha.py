"""Causal multi-head attention with rotary positions."""

import torch
import torch.nn.functional as F
from torch import nn

from .rope import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        qk_dim: int,
        num_heads: int,
        *,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()

        if qk_dim % 2 != 0:
            raise ValueError("qk_dim must be even for RoPE")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_dim = qk_dim

        self.q_size = num_heads * qk_dim
        self.k_size = num_heads * qk_dim
        self.v_size = num_heads * head_dim

        self.qkv_projection = nn.Linear(
            model_dim,
            self.q_size + self.k_size + self.v_size,
            bias=False,
        )
        self.output_projection = nn.Linear(
            self.v_size,
            model_dim,
            bias=False,
        )
        self.rope = rope if rope is not None else RotaryEmbedding(qk_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if positions is None:
            positions = self.rope(
                seq_len,
                device=x.device,
                dtype=x.dtype,
            )

        cos, sin = positions

        q, k, v = self.qkv_projection(x).split(
            (self.q_size, self.k_size, self.v_size),
            dim=-1,
        )

        q = q.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_dim,
        ).transpose(1, 2)
        k = k.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_dim,
        ).transpose(1, 2)
        v = v.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = self.rope.apply_rotary_qk(q, k, cos, sin)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        x = x.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.v_size,
        )
        return self.output_projection(x)

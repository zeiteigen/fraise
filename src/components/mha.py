import math

import torch
from torch import nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        self.dim = dim
        self.base = base

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
        compute_dtype = (
            torch.float64 if dtype == torch.float64 else torch.float32
        )

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=compute_dtype,
        )
        indices = self.frequency_indices.to(
            device=device,
            dtype=compute_dtype,
        )

        inv_freq = torch.exp(
            -indices * (math.log(self.base) / self.dim)
        )
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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        io_dim: int,
        head_dim: int,
        qk_dim: int,
        num_heads: int,
        *,
        bias: bool = False,
        attn_dropout: float = 0.0,
        rope_base: float = 10_000.0,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()

        if qk_dim % 2 != 0:
            raise ValueError("qk_dim must be even for RoPE")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_dim = qk_dim
        self.attn_dropout = attn_dropout

        self.q_size = num_heads * qk_dim
        self.k_size = num_heads * qk_dim
        self.v_size = num_heads * head_dim

        self.qkv_projection = nn.Linear(
            io_dim,
            self.q_size + self.k_size + self.v_size,
            bias=bias,
        )
        self.output_projection = nn.Linear(
            self.v_size,
            io_dim,
            bias=bias,
        )
        self.rope = (
            rope if rope is not None else RotaryEmbedding(qk_dim, base=rope_base)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if cos is None:
            cos, sin = self.rope(
                seq_len,
                device=x.device,
                dtype=x.dtype,
            )

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
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        x = x.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.v_size,
        )
        return self.output_projection(x)

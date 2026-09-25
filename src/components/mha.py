import math
from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F


def validate_right_padding_mask(padding_mask):
    """Validate a ``(B, T)`` boolean mask with padding only at each row's end.

    Call this when building or collating masks, preferably before moving them
    to an accelerator. It inspects tensor values and therefore synchronizes if
    the mask is on a GPU. ``MultiHeadAttention.forward`` trusts the declared
    layout and does not repeat this check on every call.
    """
    if not isinstance(padding_mask, torch.Tensor):
        raise TypeError("padding_mask must be a torch.Tensor")
    if padding_mask.ndim != 2:
        raise ValueError(
            "padding_mask must have shape (B, T); "
            f"got a {padding_mask.ndim}-dimensional tensor"
        )
    if padding_mask.dtype != torch.bool:
        raise TypeError(
            f"padding_mask must have dtype torch.bool; got {padding_mask.dtype}"
        )

    # True followed by False means a valid token appears after padding.
    invalid_transition = padding_mask[:, :-1] & ~padding_mask[:, 1:]
    if bool(invalid_transition.any()):
        raise ValueError(
            "right-padding masks cannot contain padding before a valid token"
        )
    return padding_mask


class RotaryEmbedding(nn.Module):
    """Apply interleaved RoPE to Q and K, leaving an odd tail unrotated."""

    def __init__(self, dim, base=10_000.0):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0:
            raise ValueError(f"dim must be a positive integer; got {dim!r}")
        if isinstance(base, bool) or not isinstance(base, Real):
            raise ValueError("base must be a finite number greater than 1")
        if not math.isfinite(base) or base <= 1.0:
            raise ValueError("base must be a finite number greater than 1")

        self.dim = int(dim)
        self.rotary_dim = self.dim - self.dim % 2
        self.base = float(base)

        # Integer indices stay precise when the parent model is cast to fp16
        # or bf16. Only this O(dim) vector is stored; sequence-sized cos/sin
        # factors are built on demand instead of caching max-length tables.
        self.register_buffer(
            "frequency_indices",
            torch.arange(0, self.rotary_dim, 2, dtype=torch.int64),
            persistent=False,
        )

    def _cos_sin(self, seq_len, device, dtype):
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        positions = torch.arange(seq_len, device=device, dtype=compute_dtype)
        frequency_indices = self.frequency_indices.to(
            device=device,
            dtype=compute_dtype,
        )
        inverse_frequencies = torch.exp(
            -frequency_indices * (math.log(self.base) / self.rotary_dim)
        )
        angles = positions[:, None] * inverse_frequencies[None, :]
        return angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]

    def _apply_rotary(self, x, cos, sin):
        rotary_values = x[..., : self.rotary_dim].to(dtype=cos.dtype)
        pairs = rotary_values.reshape(
            *rotary_values.shape[:-1],
            self.rotary_dim // 2,
            2,
        )
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        ).flatten(-2)
        rotated = rotated.to(dtype=x.dtype)

        if self.rotary_dim == self.dim:
            return rotated
        return torch.cat((rotated, x[..., self.rotary_dim :]), dim=-1)

    def forward(self, q, k):
        if self.rotary_dim == 0:
            return q, k

        cos, sin = self._cos_sin(q.shape[-2], q.device, q.dtype)
        return (
            self._apply_rotary(q, cos, sin),
            self._apply_rotary(k, cos, sin),
        )


class MultiHeadAttention(nn.Module):
    """Causal multihead self-attention with rotary positions on Q and K."""

    def __init__(
        self,
        io_dim,
        head_dim,
        qk_dim,
        num_heads,
        max_seq_len,
        bias=False,
        attn_dropout=0.0,
        rope_base=10_000.0,
    ):
        super().__init__()

        dimensions = {
            "io_dim": io_dim,
            "head_dim": head_dim,
            "qk_dim": qk_dim,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
        }
        for name, value in dimensions.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer; got {value!r}")

        if isinstance(attn_dropout, bool) or not isinstance(attn_dropout, Real):
            raise ValueError("attn_dropout must be a number in [0, 1)")
        if not math.isfinite(attn_dropout) or not 0.0 <= attn_dropout < 1.0:
            raise ValueError("attn_dropout must be a finite number in [0, 1)")
        if not isinstance(bias, bool):
            raise ValueError(f"bias must be a bool; got {bias!r}")
        if isinstance(rope_base, bool) or not isinstance(rope_base, Real):
            raise ValueError("rope_base must be a finite number greater than 1")
        if not math.isfinite(rope_base) or rope_base <= 1.0:
            raise ValueError("rope_base must be a finite number greater than 1")

        self.io_dim = int(io_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.qk_dim = int(qk_dim)
        self.attn_dropout = float(attn_dropout)
        self.rope_base = float(rope_base)

        self.q_size = self.num_heads * self.qk_dim
        self.k_size = self.num_heads * self.qk_dim
        self.v_size = self.num_heads * self.head_dim

        self.qkv_projection = nn.Linear(
            self.io_dim,
            self.q_size + self.k_size + self.v_size,
            bias=bias,
        )
        self.output_projection = nn.Linear(
            self.num_heads * self.head_dim,
            self.io_dim,
            bias=bias,
        )
        self.rotary_embedding = RotaryEmbedding(self.qk_dim, self.rope_base)

    def _validate_inputs(self, x, padding_mask, padding_layout):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor with shape (B, T, io_dim)")
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (B, T, io_dim); got a {x.ndim}-dimensional tensor"
            )

        batch_size, seq_len, io_dim = x.shape
        if batch_size == 0:
            raise ValueError("x must contain at least one batch item")
        if seq_len == 0:
            raise ValueError("x must contain at least one token")
        if io_dim != self.io_dim:
            raise ValueError(f"x's last dimension must be {self.io_dim}; got {io_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must have a floating-point dtype; got {x.dtype}")

        if padding_layout is not None and (
            not isinstance(padding_layout, str)
            or padding_layout not in ("right", "arbitrary")
        ):
            raise ValueError(
                "padding_layout must be 'right' or 'arbitrary'; "
                f"got {padding_layout!r}"
            )
        if padding_mask is None and padding_layout is not None:
            raise ValueError("padding_layout is only valid with padding_mask")
        if padding_mask is None:
            return
        if not isinstance(padding_mask, torch.Tensor):
            raise TypeError("padding_mask must be a torch.Tensor")
        if padding_layout is None:
            raise ValueError(
                "padding_layout must be specified when padding_mask is supplied"
            )
        if padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "padding_mask must have shape "
                f"({batch_size}, {seq_len}); got {tuple(padding_mask.shape)}"
            )
        if padding_mask.dtype != torch.bool:
            raise TypeError(
                f"padding_mask must have dtype torch.bool; got {padding_mask.dtype}"
            )
        if padding_mask.device != x.device:
            raise ValueError(
                "padding_mask must be on the same device as x; "
                f"got {padding_mask.device} and {x.device}"
            )

    @staticmethod
    def _sanitize_padded_inputs(x, padding_mask):
        if padding_mask is None:
            return x
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def _project_qkv(self, x):
        """Apply the fused input projection and return separate Q, K, and V."""
        qkv = self.qkv_projection(x)
        return qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

    def _split_heads(self, x, head_dim):
        """Convert ``(B, T, H * D)`` into ``(B, H, T, D)``."""
        batch_size, seq_len, _ = x.shape
        return x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            head_dim,
        ).transpose(1, 2)

    def _project_and_split_qkv(self, x):
        q, k, v = self._project_qkv(x)
        return (
            self._split_heads(q, self.qk_dim),
            self._split_heads(k, self.qk_dim),
            self._split_heads(v, self.head_dim),
        )

    def _causal_sdpa(self, q, k, v, dropout_p):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=dropout_p,
        )

    def _attention_arbitrary_padding(self, q, k, v, padding_mask, dropout_p):
        """Slow eager fallback for masks with left padding or interior holes."""
        batch_outputs = []
        for batch_idx in range(q.shape[0]):
            valid_indices = (~padding_mask[batch_idx]).nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0:
                batch_outputs.append(torch.zeros_like(v[batch_idx : batch_idx + 1]))
                continue

            row_q = q[batch_idx : batch_idx + 1].index_select(2, valid_indices)
            row_k = k[batch_idx : batch_idx + 1].index_select(2, valid_indices)
            row_v = v[batch_idx : batch_idx + 1].index_select(2, valid_indices)
            valid_output = self._causal_sdpa(row_q, row_k, row_v, dropout_p)

            row_output = torch.zeros_like(v[batch_idx : batch_idx + 1])
            batch_outputs.append(row_output.index_copy(2, valid_indices, valid_output))

        return torch.cat(batch_outputs, dim=0)

    def _attend(self, q, k, v, padding_mask, padding_layout):
        dropout_p = self.attn_dropout if self.training else 0.0
        if padding_layout in (None, "right"):
            # A right layout is a caller-validated assertion. The fast path
            # avoids reading mask values; valid queries cannot see later pads.
            return self._causal_sdpa(q, k, v, dropout_p)
        return self._attention_arbitrary_padding(
            q,
            k,
            v,
            padding_mask,
            dropout_p,
        )

    def _merge_heads(self, x):
        """Convert ``(B, H, T, D)`` into ``(B, T, H * D)``."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            num_heads * head_dim,
        )

    def _project_output(self, attention_output):
        return self.output_projection(self._merge_heads(attention_output))

    @staticmethod
    def _zero_padded_queries(output, padding_mask):
        if padding_mask is None:
            return output
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(self, x, padding_mask=None, *, padding_layout=None):
        """Apply causal self-attention to ``x`` shaped ``(B, T, io_dim)``.

        ``padding_mask`` has shape ``(B, T)`` and uses ``True`` for padding.
        If supplied, also provide ``padding_layout``:

        - ``"right"`` asserts that padding only follows valid tokens. Validate
          the mask upstream with :func:`validate_right_padding_mask`.
        - ``"arbitrary"`` supports left padding and holes through a slower eager
          fallback. It is not supported by ``torch.compile(fullgraph=True)``.

        The declared layout is trusted here to avoid device synchronization in
        the forward path. Padded inputs are zeroed before projection, and padded
        query outputs are zeroed after the output projection.
        """
        self._validate_inputs(x, padding_mask, padding_layout)
        projection_input = self._sanitize_padded_inputs(x, padding_mask)
        q, k, v = self._project_and_split_qkv(projection_input)
        q, k = self.rotary_embedding(q, k)
        attention_output = self._attend(q, k, v, padding_mask, padding_layout)
        output = self._project_output(attention_output)
        return self._zero_padded_queries(output, padding_mask)

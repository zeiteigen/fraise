from components.mha import MultiHeadAttention
from components.swiglu import SwiGLU

from torch import nn

import torch

class TransformerBlock(nn.Module):
    def __init__(
        self,
        io_dim,
        head_dim,
        qk_dim,
        num_heads,
        hidden_dim,
        max_seq_len=2048,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            io_dim,
            head_dim,
            qk_dim,
            num_heads,
            max_seq_len,
        )
        self.swiGLU = SwiGLU(io_dim, hidden_dim)

        self.layernorm1 = nn.LayerNorm(io_dim)
        self.layernorm2 = nn.LayerNorm(io_dim)

    def forward(self, x, padding_mask=None):
        # x shape: (batch_size, seq_length, io_dim)

        # Multi-Head Attention with residual connection
        mha_output = self.mha(
            self.layernorm1(x),
            padding_mask,
            padding_layout="right" if padding_mask is not None else None,
        )
        x = x + mha_output  # Residual connection

        # SwiGLU with residual connection
        swiGLU_output = self.swiGLU(self.layernorm2(x))
        x = x + swiGLU_output  # Residual connection

        return x

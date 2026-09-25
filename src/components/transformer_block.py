from components.mha import MultiHeadCausalSelfAttention
from components.swiglu import SwiGLU

from torch import nn

import torch

class TransformerBlock(nn.Module):
    def __init__(self, io_dim, head_dim, qk_dim, num_heads, hidden_dim):
        super().__init__()

        self.mha = MultiHeadCausalSelfAttention(io_dim, head_dim, qk_dim, num_heads)
        self.swiGLU = SwiGLU(io_dim, hidden_dim)

        self.layernorm1 = nn.LayerNorm(io_dim)
        self.layernorm2 = nn.LayerNorm(io_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, io_dim)

        # Multi-Head Attention with residual connection
        mha_output = self.mha(self.layernorm1(x))
        x = x + mha_output  # Residual connection

        # SwiGLU with residual connection
        swiGLU_output = self.swiGLU(self.layernorm2(x))
        x = x + swiGLU_output  # Residual connection

        return x

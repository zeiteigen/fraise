from components.mha import MultiHeadAttention, RotaryEmbedding
from components.swiglu import SwiGLU

from torch import nn

class TransformerBlock(nn.Module):
    def __init__(
        self,
        io_dim,
        head_dim,
        qk_dim,
        num_heads,
        hidden_dim,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            io_dim,
            head_dim,
            qk_dim,
            num_heads,
            rope=rope,
        )
        self.swiGLU = SwiGLU(io_dim, hidden_dim)

        self.layernorm1 = nn.LayerNorm(io_dim)
        self.layernorm2 = nn.LayerNorm(io_dim)

    def forward(self, x, cos=None, sin=None):
        # x shape: (batch_size, seq_length, io_dim)

        # Multi-Head Attention with residual connection
        mha_output = self.mha(self.layernorm1(x), cos, sin)
        x = x + mha_output  # Residual connection

        # SwiGLU with residual connection
        swiGLU_output = self.swiGLU(self.layernorm2(x))
        x = x + swiGLU_output  # Residual connection

        return x

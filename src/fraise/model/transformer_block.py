from torch import nn

from fraise.model.swiglu import SwiGLU

from .mha import MultiHeadAttention
from .rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim,
        head_dim,
        qk_dim,
        num_heads,
        hidden_dim,
        rope: RotaryEmbedding | None = None,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            model_dim,
            head_dim,
            qk_dim,
            num_heads,
            rope=rope,
        )
        self.swiglu = SwiGLU(model_dim, hidden_dim)

        self.layernorm1 = nn.LayerNorm(model_dim)
        self.layernorm2 = nn.LayerNorm(model_dim)

    def forward(self, x, positions=None):
        # x shape: (batch_size, seq_length, model_dim)

        # Multi-Head Attention with residual connection
        mha_output = self.mha(self.layernorm1(x), positions)
        x = x + mha_output  # Residual connection

        # SwiGLU with residual connection
        swiglu_output = self.swiglu(self.layernorm2(x))
        x = x + swiglu_output  # Residual connection

        return x

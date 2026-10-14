import torch
from torch import nn

from .cache import LayerCache
from .mha import MultiHeadAttention
from .rope import RotaryEmbedding
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        qk_dim: int,
        num_heads: int,
        hidden_dim: int,
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

    def forward(
        self,
        x: torch.Tensor,
        positions: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: LayerCache | None = None,
    ) -> torch.Tensor:
        # x shape: (batch_size, seq_length, model_dim); pre-norm residual blocks.
        x = x + self.mha(self.layernorm1(x), positions, cache)
        return x + self.swiglu(self.layernorm2(x))

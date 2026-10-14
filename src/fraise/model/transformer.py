import torch
from torch import nn

from .cache import KVCache
from .rope import RotaryEmbedding
from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        qk_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.rope = RotaryEmbedding(qk_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    head_dim,
                    qk_dim,
                    num_heads,
                    hidden_dim,
                    rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        # x shape: (batch_size, seq_length, model_dim)
        positions = self.rope(
            x.size(1),
            device=x.device,
            dtype=x.dtype,
            offset=0 if cache is None else len(cache),
        )
        for index, layer in enumerate(self.layers):
            x = layer(x, positions, None if cache is None else cache.layers[index])
        return x

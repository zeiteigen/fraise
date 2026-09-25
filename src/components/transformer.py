from components.mha import RotaryEmbedding
from components.transformer_block import TransformerBlock
from torch import nn

class Transformer(nn.Module):
    def __init__(
        self,
        io_dim,
        head_dim,
        qk_dim,
        num_heads,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        self.rope = RotaryEmbedding(qk_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    io_dim,
                    head_dim,
                    qk_dim,
                    num_heads,
                    hidden_dim,
                    rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, io_dim)

        cos, sin = self.rope(
            x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        for layer in self.layers:
            x = layer(x, cos, sin)

        return x

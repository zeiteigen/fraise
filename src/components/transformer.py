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
        max_seq_len=2048,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    io_dim,
                    head_dim,
                    qk_dim,
                    num_heads,
                    hidden_dim,
                    max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, padding_mask=None):
        # x shape: (batch_size, seq_length, io_dim)

        for layer in self.layers:
            x = layer(x, padding_mask)

        return x

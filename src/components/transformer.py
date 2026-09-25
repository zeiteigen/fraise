from components.transformer_block import TransformerBlock
from torch import nn
import torch

class Transformer(nn.Module):
    def __init__(self, io_dim, head_dim, qk_dim, num_heads, hidden_dim, num_layers):
        super().__init__()

        self.layers = nn.Sequential([
            TransformerBlock(io_dim, head_dim, qk_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_length, io_dim)

        for layer in self.layers:
            x = layer(x)

        return x
import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, io_dim, hidden_dim):
        super().__init__()

        self.value = nn.Linear(io_dim, hidden_dim)
        self.gate = nn.Linear(io_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, io_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, io_dim)

        value = self.value(x)          # (..., hidden_dim)
        gate = F.silu(self.gate(x))    # (..., hidden_dim)

        hidden = value * gate          # elementwise gating

        return self.output(hidden)     # (..., io_dim)
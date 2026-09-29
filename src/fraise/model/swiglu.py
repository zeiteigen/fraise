import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super().__init__()

        self.value = nn.Linear(model_dim, hidden_dim)
        self.gate = nn.Linear(model_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, model_dim)

        value = self.value(x)  # (..., hidden_dim)
        gate = F.silu(self.gate(x))  # (..., hidden_dim)

        hidden = value * gate  # elementwise gating

        return self.output(hidden)  # (..., model_dim)

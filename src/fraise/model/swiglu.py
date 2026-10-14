import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int):
        super().__init__()

        self.value = nn.Linear(model_dim, hidden_dim)
        self.gate = nn.Linear(model_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length, model_dim)
        value = self.value(x)  # (..., hidden_dim)
        gate = F.silu(self.gate(x))  # (..., hidden_dim)
        return self.output(value * gate)  # (..., model_dim)

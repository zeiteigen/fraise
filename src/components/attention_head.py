from torch import nn
import torch



class AttentionHead(nn.Module):
    def __init__(self, input_dim, qk_dim, head_dim):
        """ USE MULTIHEAD ATTENTION INSTEAD OF THIS CLASS. THIS IS ONLY FOR REFERENCE."""
        super().__init__()

        self.qk_dim = qk_dim
        self.head_dim = head_dim

        self.query = nn.Linear(input_dim, qk_dim)
        self.key = nn.Linear(input_dim, qk_dim)
        self.value = nn.Linear(input_dim, head_dim)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)

        Q = self.query(x)  # (batch_size, seq_length, qk_dim)
        K = self.key(x)    # (batch_size, seq_length, qk_dim)
        V = self.value(x)  # (batch_size, seq_length, head_dim)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / (self.qk_dim ** 0.5)
        # scores: (batch_size, seq_length, seq_length)

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        # output: (batch_size, seq_length, head_dim)

        return output
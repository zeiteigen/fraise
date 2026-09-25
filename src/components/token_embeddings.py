import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:
                Shape: [batch, seq]
                dtype: torch.long

        Returns:
            embeddings:
                Shape: [batch, seq, d_model]
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids with shape [batch, seq], "
                f"got {tuple(input_ids.shape)}"
            )

        return self.embedding(input_ids)
"""Construct and run the Fraise baseline transformer."""

from torch import nn

from fraise.config.model import ModelConfig

from .transformer import Transformer


class FraiseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.residual_stream_dim
        )
        self.transformer = Transformer(
            config.residual_stream_dim,
            config.head_dim,
            config.qk_dim,
            config.num_heads,
            config.swiglu_up_dim,
            config.num_layers,
        )
        self.lm_head = nn.Linear(config.residual_stream_dim, config.vocab_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len {self.max_seq_len}")
        return self.lm_head(self.transformer(self.token_embedding(input_ids)))

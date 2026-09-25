
from components.transformer import Transformer
from components.lm_head import LMHead
from components.token_embeddings import TokenEmbedding

from torch import nn


from typing import TypedDict

class ModelConfig(TypedDict):
    vocab_size: int
    residual_stream_dim: int
    head_dim: int
    qk_dim: int
    num_heads: int
    swiglu_up_dim: int
    num_layers: int
    max_seq_len: int

class FraiseModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        vocab_size = config["vocab_size"]
        residual_stream_dim = config["residual_stream_dim"]
        head_dim = config["head_dim"]
        qk_dim = config["qk_dim"]
        num_heads = config["num_heads"]
        swiglu_up_dim = config["swiglu_up_dim"]
        num_layers = config["num_layers"]
        max_seq_len = config.get("max_seq_len", 2048)

        self.token_embedding = TokenEmbedding(vocab_size, residual_stream_dim)
        self.transformer = Transformer(
            residual_stream_dim,
            head_dim,
            qk_dim,
            num_heads,
            swiglu_up_dim,
            num_layers,
            max_seq_len,
        )
        self.lm_head = LMHead(residual_stream_dim, vocab_size)

    def forward(self, input_ids, padding_mask=None):
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.transformer(hidden_states, padding_mask)
        return self.lm_head(hidden_states)

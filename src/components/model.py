
from components.transformer import Transformer
from components.lm_head import LMHead
from components.token_embeddings import TokenEmbedding

from torch import nn


from typing import TypedDict

class ModelConfig(TypedDict):
    vocab_size: int
    io_dim: int
    head_dim: int
    qk_dim: int
    num_heads: int
    hidden_dim: int
    num_layers: int

class FraiseModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        vocab_size = config["vocab_size"]
        io_dim = config["io_dim"]
        head_dim = config["head_dim"]
        qk_dim = config["qk_dim"]
        num_heads = config["num_heads"]
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]

        self.model = nn.Sequential(
            TokenEmbedding(vocab_size, io_dim),
            Transformer(io_dim, head_dim, qk_dim, num_heads, hidden_dim, num_layers),
            LMHead(io_dim, vocab_size),
        )

    def forward(self, input_ids):
        return self.model(input_ids)
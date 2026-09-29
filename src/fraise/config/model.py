"""The model's dimensions and supported options."""

from dataclasses import dataclass, fields

from .values import integer


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    residual_stream_dim: int
    head_dim: int
    qk_dim: int
    num_heads: int
    swiglu_up_dim: int
    num_layers: int
    max_seq_len: int = 2048
    tie_word_embeddings: bool = False

    def __post_init__(self):
        for field in fields(self):
            if field.name != "tie_word_embeddings":
                integer(field.name, getattr(self, field.name))
        if type(self.tie_word_embeddings) is not bool:
            raise TypeError("tie_word_embeddings must be a bool")
        if self.qk_dim % 2:
            raise ValueError("qk_dim must be even for RoPE")

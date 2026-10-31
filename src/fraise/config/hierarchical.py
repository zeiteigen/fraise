"""Dimensions of the asymmetric Fraise model: macro input, base-token output."""

from dataclasses import dataclass, fields

from .values import integer

MACRO_EMBEDDINGS = ("compositional", "independent")


@dataclass(frozen=True)
class FraiseConfig:
    """``vocab_size`` is the macro (input) vocabulary of the global transformer;
    ``base_vocab_size`` is the base (output) vocabulary of the local decoder.

    ``max_macro_length`` bounds base tokens per macro, the depth of the
    decoding trie. ``macro_embedding`` selects compositional macro inputs
    (constituent base embeddings plus a ``macro_residual_dim`` learned residual;
    0 disables residuals) or one independent embedding per macro.
    """

    vocab_size: int
    base_vocab_size: int
    max_macro_length: int
    residual_stream_dim: int
    head_dim: int
    qk_dim: int
    num_heads: int
    swiglu_up_dim: int
    num_layers: int
    max_seq_len: int = 2048
    macro_embedding: str = "compositional"
    macro_residual_dim: int = 32
    local_dim: int = 512
    local_num_layers: int = 2
    local_swiglu_up_dim: int = 1024
    architecture: str = "fraise"

    def __post_init__(self):
        for field in fields(self):
            if field.name in {"architecture", "macro_embedding", "macro_residual_dim"}:
                continue
            integer(field.name, getattr(self, field.name))
        integer("macro_residual_dim", self.macro_residual_dim, 0)
        if self.architecture != "fraise":
            raise ValueError("FraiseConfig only supports architecture 'fraise'")
        if self.macro_embedding not in MACRO_EMBEDDINGS:
            raise ValueError(f"macro_embedding must be one of {MACRO_EMBEDDINGS}")
        if self.macro_embedding == "independent" and self.macro_residual_dim:
            raise ValueError(
                "independent macro embeddings have no residual; set macro_residual_dim to 0"
            )
        if self.base_vocab_size > self.vocab_size:
            raise ValueError("base_vocab_size cannot exceed the macro vocab_size")
        if self.qk_dim % 2:
            raise ValueError("qk_dim must be even for RoPE")

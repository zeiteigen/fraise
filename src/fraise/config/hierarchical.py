"""Dimensions of the asymmetric Fraise model: macro input, base-token output."""

from dataclasses import dataclass, fields

from .hierarchy import HierarchyConfig
from .values import integer

MACRO_EMBEDDINGS = ("compositional", "independent", "hierarchical")


@dataclass(frozen=True)
class FraiseConfig:
    """``vocab_size`` is the macro (input) vocabulary of the global transformer;
    ``base_vocab_size`` is the base (output) vocabulary of the local decoder.

    ``max_macro_length`` bounds base tokens per macro, the depth of the
    decoding trie. ``macro_embedding`` selects compositional macro inputs
    (constituent base embeddings plus a ``macro_residual_dim`` learned residual;
    0 disables residuals), one independent embedding per macro, or a
    multi-level ``embedding_hierarchy`` (which then owns all residuals, so
    ``macro_residual_dim`` must be 0).
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
    embedding_hierarchy: HierarchyConfig | None = None
    architecture: str = "fraise"

    def __post_init__(self):
        skipped = {
            "architecture",
            "macro_embedding",
            "macro_residual_dim",
            "embedding_hierarchy",
        }
        for field in fields(self):
            if field.name in skipped:
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
        if self.embedding_hierarchy is not None:
            object.__setattr__(
                self,
                "embedding_hierarchy",
                HierarchyConfig.from_value(self.embedding_hierarchy),
            )
        hierarchical = self.macro_embedding == "hierarchical"
        if hierarchical != (self.embedding_hierarchy is not None):
            raise ValueError(
                "macro_embedding 'hierarchical' requires embedding_hierarchy, "
                "and embedding_hierarchy requires macro_embedding 'hierarchical'"
            )
        if hierarchical:
            hierarchy = self.embedding_hierarchy
            if self.macro_residual_dim:
                raise ValueError(
                    "hierarchical embeddings configure residuals per level; "
                    "set macro_residual_dim to 0"
                )
            if hierarchy.levels[0].vocab_size != self.base_vocab_size:
                raise ValueError("hierarchy level 0 must match base_vocab_size")
            if hierarchy.levels[-1].vocab_size < self.vocab_size:
                raise ValueError(
                    "the top hierarchy level must contain every input token"
                )
            if (
                hierarchy.source == "tokenizer_merges"
                and hierarchy.levels[-1].vocab_size != self.vocab_size
            ):
                raise ValueError(
                    "a tokenizer_merges hierarchy's top level must equal vocab_size"
                )
        if self.base_vocab_size > self.vocab_size:
            raise ValueError("base_vocab_size cannot exceed the macro vocab_size")
        if self.qk_dim % 2:
            raise ValueError("qk_dim must be even for RoPE")

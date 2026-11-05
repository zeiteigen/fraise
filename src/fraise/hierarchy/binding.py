"""Resolve a configured hierarchy and check it against the model and tokenizer."""

import numpy as np

from fraise.config.hierarchy import HierarchyConfig

from .construction import hierarchy_from_macro_tokenizer
from .definition import EmbeddingHierarchy, file_sha256


def resolve_hierarchy(hierarchy: HierarchyConfig, tokenizer) -> EmbeddingHierarchy:
    """Build or load the definition a configuration selects (memoized)."""
    if hierarchy.source == "file":
        key = ("file", hierarchy.path, hierarchy.sha256)
    else:
        key = (
            "tokenizer_merges",
            hierarchy.level_sizes,
            tuple(level.residual_dim for level in hierarchy.levels[1:]),
        )
    cache = _cache(tokenizer)
    if key not in cache:
        if hierarchy.source == "file":
            actual = file_sha256(hierarchy.path)
            if actual != hierarchy.sha256:
                raise ValueError(
                    f"hierarchy file {hierarchy.path} has sha256 {actual}; "
                    f"the model configuration expects {hierarchy.sha256}"
                )
            cache[key] = EmbeddingHierarchy.load(hierarchy.path)
        else:
            if getattr(tokenizer, "merges", None) is None:
                raise ValueError("a tokenizer_merges hierarchy needs a macro tokenizer")
            cache[key] = hierarchy_from_macro_tokenizer(
                tokenizer,
                hierarchy.level_sizes,
                [level.residual_dim for level in hierarchy.levels[1:]],
            )
    return cache[key]


def check_hierarchy(
    definition: EmbeddingHierarchy, hierarchy: HierarchyConfig, vocab_size: int
) -> None:
    """Require the definition to fit the configured parameter shapes exactly."""
    if definition.level_sizes != hierarchy.level_sizes:
        raise ValueError(
            f"hierarchy level sizes {list(definition.level_sizes)} differ from the "
            f"configured {list(hierarchy.level_sizes)}"
        )
    for index, level in enumerate(hierarchy.levels[1:], start=1):
        longest = definition.max_composition_length(index)
        if longest > level.max_composition_length:
            raise ValueError(
                f"level {index} composes up to {longest} children; raise its "
                f"max_composition_length (currently {level.max_composition_length})"
            )
        start, end = definition.owned_range(index)
        widths = definition.residual_widths[start:end]
        if int(widths.max(initial=0)) > level.residual_dim:
            raise ValueError(
                f"level {index} has residuals of width {int(widths.max())}; raise "
                f"its residual_dim (currently {level.residual_dim})"
            )
        if int(widths.sum()) != hierarchy.residual_parameters(index):
            raise ValueError(
                f"level {index} residual widths sum to {int(widths.sum())}, but the "
                f"configuration allocates {hierarchy.residual_parameters(index)}"
            )
    definition.input_node_ids(vocab_size)


def check_tokenizer_hierarchy(definition: EmbeddingHierarchy, tokenizer) -> None:
    """Every input token must expand to the base tokens its tokenizer emits."""
    table = getattr(tokenizer, "decompositions", None)
    if table is None:
        vocab = tokenizer.get_vocab_size()
        table, lengths = np.arange(vocab)[:, None], np.ones(vocab, dtype=np.int64)
    else:
        lengths = tokenizer.unit_lengths
    definition.check_against_decompositions(table, lengths)


def _cache(tokenizer) -> dict:
    try:
        return tokenizer.__dict__.setdefault("_fraise_hierarchies", {})
    except AttributeError:  # tokenizers.Tokenizer has no instance dictionary.
        return {}

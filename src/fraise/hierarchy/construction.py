"""Build simple valid hierarchies from existing tokenizers (for testing).

This is deliberately not a vocabulary-selection algorithm: intermediate levels
are merge-order prefixes of a macro tokenizer. Macro ``m`` was created by
merging two earlier macros, so its children at level ``l`` are found by
expanding its merge tree until every part belongs to level ``l - 1``.
"""

from collections.abc import Sequence

import numpy as np

from .definition import EmbeddingHierarchy


def hierarchy_from_macro_tokenizer(
    tokenizer,
    level_sizes: Sequence[int],
    residual_dims: int | Sequence[int] = 32,
    *,
    level_names: Sequence[str] | None = None,
) -> EmbeddingHierarchy:
    """Nested merge-prefix levels with uniform residual widths per level.

    ``level_sizes`` starts at the base vocabulary and ends at the macro
    vocabulary. ``residual_dims`` gives each composed level's width (one value
    for all); level 1's width also applies to atoms, as in the two-level model.
    """
    sizes = [int(size) for size in level_sizes]
    if getattr(tokenizer, "merges", None) is None:
        raise ValueError("a merge-prefix hierarchy needs a macro tokenizer")
    base = tokenizer.base_vocab_size
    vocab = tokenizer.get_vocab_size()
    if len(sizes) < 2 or sizes[0] != base or sizes[-1] != vocab:
        raise ValueError(
            f"level sizes must run from the base vocabulary ({base}) to the "
            f"macro vocabulary ({vocab})"
        )
    if any(upper <= lower for lower, upper in zip(sizes, sizes[1:])):
        raise ValueError("level sizes must strictly increase")
    if isinstance(residual_dims, int):
        residual_dims = [residual_dims] * (len(sizes) - 1)
    residual_dims = [int(width) for width in residual_dims]
    if len(residual_dims) == 1:
        residual_dims *= len(sizes) - 1
    if len(residual_dims) != len(sizes) - 1 or min(residual_dims) < 0:
        raise ValueError("supply one non-negative residual width per composed level")
    merges = tokenizer.merges
    compositions = []
    for lower, upper in zip(sizes, sizes[1:]):
        # Merge k creates macro base + k; earlier IDs are complete before later ones.
        frontier: dict[int, list[int]] = {}

        def expand(node: int) -> list[int]:
            return [node] if node < lower else frontier[node]

        for node in range(lower, upper):
            left, right = merges[node - base]
            frontier[node] = expand(left) + expand(right)
        compositions.append([frontier[node] for node in range(lower, upper)])
    widths = np.empty(sizes[-1], dtype=np.int64)
    widths[: sizes[1]] = residual_dims[0]
    for level in range(2, len(sizes)):
        widths[sizes[level - 1] : sizes[level]] = residual_dims[level - 1]
    return EmbeddingHierarchy(
        sizes,
        compositions,
        widths,
        level_names=level_names,
        metadata={"source": "tokenizer_merges"},
    )


def two_level_hierarchy(decompositions, lengths, base_vocab_size, residual_dim):
    """The original 8k -> 64k model as a hierarchy: macros compose base tokens."""
    decompositions = np.asarray(decompositions, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    vocab = len(lengths)
    width = np.arange(decompositions.shape[1])
    rows = np.where(
        width[None, :] < lengths[base_vocab_size:, None],
        decompositions[base_vocab_size:],
        -1,
    )
    return EmbeddingHierarchy(
        (base_vocab_size, vocab),
        (rows,),
        np.full(vocab, residual_dim, dtype=np.int64),
        metadata={"source": "two_level"},
    )

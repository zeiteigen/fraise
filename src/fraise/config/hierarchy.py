"""Shapes of a multi-level embedding hierarchy (8k -> 16k -> 64k, ...).

The configuration fixes every parameter shape, so parameter counts are exact
without allocating a model. Which tokens compose which lives in an
``EmbeddingHierarchy`` definition (``fraise.hierarchy``) that is bound into
model buffers, so checkpoints are self-contained.

Level ``l`` holds node IDs ``0 .. vocab_size_l - 1``; vocabularies are nested,
so a node introduced at level ``l`` also exists at every higher level. Level 0
is the atom (base-token) table. A node's residual is owned by exactly one level:
the level that introduces it, except atoms, whose residuals belong to level 1
(the first composition), as in the two-level model.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields

from .values import integer

COMPOSITION_SHARING = ("per_level", "shared")
HIERARCHY_SOURCES = ("tokenizer_merges", "file")


@dataclass(frozen=True)
class HierarchyLevelConfig:
    """``vocab_size`` counts every node of this level (nested, cumulative).

    ``max_composition_length`` bounds the children of a node introduced here.
    ``residual_dim`` is the widest residual owned by this level and the number
    of projection rows it uses; individual tokens may be narrower, including
    zero. ``residual_parameters`` is the sum of owned widths; ``None`` means
    every owned node has exactly ``residual_dim``.
    """

    vocab_size: int
    max_composition_length: int = 0
    residual_dim: int = 0
    residual_parameters: int | None = None

    def __post_init__(self):
        integer("vocab_size", self.vocab_size)
        integer("max_composition_length", self.max_composition_length, 0)
        integer("residual_dim", self.residual_dim, 0)
        if self.residual_parameters is not None:
            integer("residual_parameters", self.residual_parameters, 0)


@dataclass(frozen=True)
class HierarchyConfig:
    """``source`` selects how the definition is supplied when a run starts.

    ``"tokenizer_merges"`` derives it from a macro tokenizer's merge tree, using
    the level sizes as merge-order prefixes (uniform residual widths).
    ``"file"`` loads an externally generated definition from ``path`` and
    verifies ``sha256`` (of the file bytes). ``composition`` shares one
    composition network across levels or gives each level its own.
    """

    levels: tuple[HierarchyLevelConfig, ...]
    composition: str = "per_level"
    source: str = "tokenizer_merges"
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self):
        levels = tuple(
            level
            if isinstance(level, HierarchyLevelConfig)
            else HierarchyLevelConfig(**_known(HierarchyLevelConfig, level, "level"))
            for level in self.levels
        )
        object.__setattr__(self, "levels", levels)
        if len(levels) < 2:
            raise ValueError("an embedding hierarchy needs at least two levels")
        if levels[0].max_composition_length or levels[0].residual_dim:
            raise ValueError(
                "level 0 holds atoms: its max_composition_length and residual_dim "
                "must be 0 (atom residuals belong to level 1)"
            )
        if levels[0].residual_parameters not in (None, 0):
            raise ValueError("level 0 owns no residual parameters")
        for index in range(1, len(levels)):
            level, below = levels[index], levels[index - 1]
            if level.vocab_size <= below.vocab_size:
                raise ValueError("level vocab sizes must strictly increase")
            if level.max_composition_length < 1:
                raise ValueError(f"level {index} needs max_composition_length >= 1")
            owned = self.owned_nodes(index)
            if (
                level.residual_parameters is not None
                and level.residual_parameters > owned * level.residual_dim
            ):
                raise ValueError(
                    f"level {index} residual_parameters exceeds its {owned} owned "
                    f"nodes x residual_dim {level.residual_dim}"
                )
        if self.composition not in COMPOSITION_SHARING:
            raise ValueError(f"composition must be one of {COMPOSITION_SHARING}")
        if self.source not in HIERARCHY_SOURCES:
            raise ValueError(f"hierarchy source must be one of {HIERARCHY_SOURCES}")
        if self.source == "file":
            if not isinstance(self.path, str) or not self.path:
                raise ValueError("a file hierarchy needs a path")
            if not isinstance(self.sha256, str) or not re.fullmatch(
                "[0-9a-f]{64}", self.sha256
            ):
                raise ValueError("a file hierarchy needs its lowercase hex sha256")
        else:
            if self.path is not None or self.sha256 is not None:
                raise ValueError("path and sha256 apply only to source 'file'")
            if any(level.residual_parameters is not None for level in levels):
                raise ValueError(
                    "tokenizer_merges hierarchies use uniform residual widths; "
                    "omit residual_parameters"
                )

    @classmethod
    def from_value(cls, value) -> HierarchyConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("embedding_hierarchy must be a JSON object")
        values = _known(cls, value, "embedding_hierarchy")
        if not isinstance(values.get("levels"), (list, tuple)):
            raise TypeError("embedding_hierarchy.levels must be a list")
        return cls(**values)

    @property
    def level_sizes(self) -> tuple[int, ...]:
        return tuple(level.vocab_size for level in self.levels)

    def owned_start(self, index: int) -> int:
        """First node whose residual level ``index`` owns (atoms go to level 1)."""
        return 0 if index == 1 else self.levels[index - 1].vocab_size

    def owned_nodes(self, index: int) -> int:
        if index == 0:
            return 0
        return self.levels[index].vocab_size - self.owned_start(index)

    def new_nodes(self, index: int) -> int:
        return self.levels[index].vocab_size - self.levels[index - 1].vocab_size

    def residual_parameters(self, index: int) -> int:
        level = self.levels[index]
        if index == 0:
            return 0
        if level.residual_parameters is not None:
            return level.residual_parameters
        return self.owned_nodes(index) * level.residual_dim

    def composers(self) -> list[tuple[int, int]]:
        """``(max_composition_length, residual_dim)`` per composition network."""
        shapes = [
            (level.max_composition_length, level.residual_dim)
            for level in self.levels[1:]
        ]
        if self.composition == "shared":
            return [(max(k for k, _ in shapes), max(r for _, r in shapes))]
        return shapes

    def composer_index(self, index: int) -> int:
        return 0 if self.composition == "shared" else index - 1


def _known(cls, values: Mapping, name: str) -> dict:
    names = {field.name for field in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {name} settings: {sorted(unknown)}")
    return dict(values)

"""An externally supplied embedding hierarchy: which nodes compose which.

Nodes live in one ID space ordered by the level that introduces them. Level
``l`` contains nodes ``0 .. level_sizes[l] - 1`` (vocabularies are nested), so
a node exists at every level from its introduction upward and owns exactly one
residual. Level 0 nodes are atoms (base tokens).

For every level ``l >= 1``, ``compositions[l - 1]`` lists the ordered children
of the *new* nodes ``level_sizes[l-1] .. level_sizes[l] - 1``; children are
nodes of level ``l - 1``. Rows are padded with ``-1`` after their last child.
A node carried over from a lower level is recomposed from itself alone.

``residual_widths[n]`` is node ``n``'s residual width (0 allowed).
``input_nodes[t]`` maps input token ``t`` to its node; ``None`` means the
identity, which requires the top level to equal the input vocabulary.
Intermediate nodes need not be input tokens: only the input tokenizer decides
global transformer positions.

On disk the definition is ``.npz`` (compact, for large vocabularies) or
``.json`` (readable, for small or hand-written ones); see ``save``.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "fraise-embedding-hierarchy"
FORMAT_VERSION = 1


@dataclass
class EmbeddingHierarchy:
    level_sizes: tuple[int, ...]
    compositions: tuple[np.ndarray, ...]
    residual_widths: np.ndarray
    input_nodes: np.ndarray | None = None
    level_names: tuple[str, ...] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.level_sizes = tuple(int(size) for size in self.level_sizes)
        self.compositions = tuple(
            _padded(rows, index + 1) for index, rows in enumerate(self.compositions)
        )
        self.residual_widths = np.asarray(self.residual_widths, dtype=np.int64)
        if self.input_nodes is not None:
            self.input_nodes = np.asarray(self.input_nodes, dtype=np.int64)
        if self.level_names is not None:
            self.level_names = tuple(str(name) for name in self.level_names)
        self.validate()

    # Structure ----------------------------------------------------------

    @property
    def num_levels(self) -> int:
        return len(self.level_sizes)

    @property
    def num_nodes(self) -> int:
        return self.level_sizes[-1]

    def level_name(self, level: int) -> str:
        if self.level_names is not None:
            return self.level_names[level]
        return f"L{level} ({_short(self.level_sizes[level])})"

    def validate(self) -> None:
        sizes = self.level_sizes
        if len(sizes) < 2:
            raise ValueError("an embedding hierarchy needs at least two levels")
        if sizes[0] < 1 or any(b <= a for a, b in zip(sizes, sizes[1:])):
            raise ValueError("level sizes must be positive and strictly increase")
        if len(self.compositions) != len(sizes) - 1:
            raise ValueError("supply one composition table per level above 0")
        for level, rows in enumerate(self.compositions, start=1):
            if rows.shape[0] != sizes[level] - sizes[level - 1]:
                raise ValueError(f"level {level} needs one composition per new node")
            lengths = self.composition_lengths(level)
            if lengths.min(initial=1) < 1:
                raise ValueError(f"level {level} has a node without children")
            width = np.arange(rows.shape[1])
            present = width[None, :] < lengths[:, None]
            if np.any((rows >= 0) != present):
                raise ValueError(f"level {level} children must precede -1 padding")
            if np.any(rows[present] >= sizes[level - 1]):
                raise ValueError(
                    f"level {level} children must be nodes of level {level - 1}"
                )
        if self.residual_widths.shape != (self.num_nodes,):
            raise ValueError("residual_widths needs one width per node")
        if self.residual_widths.min(initial=0) < 0:
            raise ValueError("residual widths cannot be negative")
        if self.input_nodes is not None:
            nodes = self.input_nodes
            if nodes.ndim != 1 or nodes.min(initial=0) < 0:
                raise ValueError("input_nodes must be non-negative node IDs")
            if nodes.max(initial=0) >= self.num_nodes:
                raise ValueError("input_nodes refers to an undefined node")
            if len(np.unique(nodes)) != len(nodes):
                raise ValueError("two input tokens cannot share one node")
        if self.level_names is not None and len(self.level_names) != len(sizes):
            raise ValueError("level_names needs one name per level")

    def composition_lengths(self, level: int) -> np.ndarray:
        return (self.compositions[level - 1] >= 0).sum(axis=1)

    def max_composition_length(self, level: int) -> int:
        return int(self.composition_lengths(level).max(initial=0))

    def introduced_at(self, node: int) -> int:
        return int(np.searchsorted(self.level_sizes, node, side="right"))

    def owner_level(self, node: int) -> int:
        """The level whose composition adds this node's residual."""
        return max(1, self.introduced_at(node))

    def owned_range(self, level: int) -> tuple[int, int]:
        start = 0 if level == 1 else self.level_sizes[level - 1]
        return start, self.level_sizes[level]

    def children(self, node: int, level: int) -> list[int]:
        """Ordered level ``level - 1`` children of ``node`` at ``level``."""
        if not 1 <= level < self.num_levels or node >= self.level_sizes[level]:
            raise ValueError(f"node {node} is not composed at level {level}")
        start = self.level_sizes[level - 1]
        if node < start:
            return [node]
        row = self.compositions[level - 1][node - start]
        return row[row >= 0].tolist()

    def input_node_ids(self, vocab_size: int) -> np.ndarray:
        if self.input_nodes is None:
            if vocab_size != self.num_nodes:
                raise ValueError(
                    "without input_nodes the top level must equal the input vocabulary"
                )
            return np.arange(vocab_size, dtype=np.int64)
        if len(self.input_nodes) != vocab_size:
            raise ValueError("input_nodes needs one node per input token")
        return self.input_nodes

    def atom_expansions(self) -> tuple[np.ndarray, np.ndarray]:
        """Every node's ordered atoms ``[nodes, width]`` (0-padded) and counts."""
        sizes = self.level_sizes
        counts = np.ones(self.num_nodes, dtype=np.int64)
        widths = [1]
        for level in range(1, self.num_levels):
            rows = self.compositions[level - 1]
            present = rows >= 0
            child_counts = np.where(present, counts[np.maximum(rows, 0)], 0)
            counts[sizes[level - 1] : sizes[level]] = child_counts.sum(axis=1)
            widths.append(int(counts[: sizes[level]].max()))
        table = np.zeros((self.num_nodes, max(widths)), dtype=np.int64)
        table[: sizes[0], 0] = np.arange(sizes[0])
        for level in range(1, self.num_levels):
            rows = self.compositions[level - 1]
            new = slice(sizes[level - 1], sizes[level])
            block = np.zeros((rows.shape[0], table.shape[1]), dtype=np.int64)
            offset = np.zeros(rows.shape[0], dtype=np.int64)
            row_index = np.arange(rows.shape[0])
            for slot in range(rows.shape[1]):
                child = rows[:, slot]
                present = child >= 0
                safe = np.maximum(child, 0)
                child_count = np.where(present, counts[safe], 0)
                for position in range(widths[level - 1]):
                    take = present & (position < child_count)
                    block[row_index[take], offset[take] + position] = table[
                        safe[take], position
                    ]
                offset += child_count
            table[new] = block
        return table, counts

    def check_against_decompositions(self, decompositions, lengths) -> None:
        """Require each input token's atoms to spell its tokenizer decomposition."""
        lengths = np.asarray(lengths, dtype=np.int64)
        decompositions = np.asarray(decompositions, dtype=np.int64)
        nodes = self.input_node_ids(len(lengths))
        table, counts = self.atom_expansions()
        mismatched = counts[nodes] != lengths
        width = min(table.shape[1], decompositions.shape[1])
        present = np.arange(width)[None, :] < lengths[:, None]
        differs = (table[nodes, :width] != decompositions[:, :width]) & present
        mismatched |= differs.any(axis=1)
        if mismatched.any():
            token = int(np.flatnonzero(mismatched)[0])
            raise ValueError(
                f"hierarchy expands input token {token} to "
                f"{table[nodes[token], : counts[nodes[token]]].tolist()}, but the "
                f"tokenizer decomposes it to "
                f"{decompositions[token, : lengths[token]].tolist()}"
            )

    # Accounting ---------------------------------------------------------

    def level_summary(self) -> list[dict]:
        """Per-level node counts, composition lengths, and owned residuals."""
        records = []
        for level, size in enumerate(self.level_sizes):
            record = {
                "level": level,
                "name": self.level_name(level),
                "vocab_size": size,
                "new_nodes": size - (self.level_sizes[level - 1] if level else 0),
            }
            if level:
                start, end = self.owned_range(level)
                widths = self.residual_widths[start:end]
                lengths = self.composition_lengths(level)
                record |= {
                    "max_composition_length": int(lengths.max()),
                    "mean_composition_length": float(lengths.mean()),
                    "owned_nodes": end - start,
                    "residual_dim": int(widths.max(initial=0)),
                    "residual_parameters": int(widths.sum()),
                    "residual_width_histogram": {
                        int(width): count
                        for width, count in sorted(Counter(widths.tolist()).items())
                    },
                }
            records.append(record)
        return records

    def config_levels(self) -> list[dict]:
        """The ``embedding_hierarchy.levels`` model configuration for this file."""
        levels = [{"vocab_size": self.level_sizes[0]}]
        for record in self.level_summary()[1:]:
            levels.append(
                {
                    "vocab_size": record["vocab_size"],
                    "max_composition_length": record["max_composition_length"],
                    "residual_dim": record["residual_dim"],
                    "residual_parameters": record["residual_parameters"],
                }
            )
        return levels

    # Serialization ------------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Write ``.npz`` or ``.json`` and return the file's sha256."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "level_sizes": list(self.level_sizes),
            "level_names": list(self.level_names) if self.level_names else None,
            "metadata": self.metadata,
        }
        temporary = path.with_name(path.name + ".tmp")
        if path.suffix == ".npz":
            arrays = {
                f"compositions_{level}": rows.astype(np.int32)
                for level, rows in enumerate(self.compositions, start=1)
            }
            arrays["residual_widths"] = self.residual_widths.astype(np.int32)
            if self.input_nodes is not None:
                arrays["input_nodes"] = self.input_nodes
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, header=np.array(json.dumps(header)), **arrays)
        elif path.suffix == ".json":
            header |= {
                "compositions": [
                    [row[row >= 0].tolist() for row in rows]
                    for rows in self.compositions
                ],
                "residual_widths": self.residual_widths.tolist(),
                "input_nodes": None
                if self.input_nodes is None
                else self.input_nodes.tolist(),
            }
            temporary.write_text(json.dumps(header, separators=(",", ":")))
        else:
            raise ValueError("hierarchy files must end in .npz or .json")
        temporary.replace(path)
        return file_sha256(path)

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingHierarchy:
        path = Path(path).expanduser()
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                header = _header(json.loads(str(archive["header"])))
                levels = len(header["level_sizes"])
                compositions = [
                    archive[f"compositions_{level}"] for level in range(1, levels)
                ]
                widths = archive["residual_widths"]
                inputs = archive["input_nodes"] if "input_nodes" in archive else None
        elif path.suffix == ".json":
            header = _header(json.loads(path.read_text(encoding="utf-8")))
            compositions = header["compositions"]
            widths, inputs = header["residual_widths"], header.get("input_nodes")
        else:
            raise ValueError("hierarchy files must end in .npz or .json")
        return cls(
            header["level_sizes"],
            compositions,
            widths,
            inputs,
            header.get("level_names"),
            header.get("metadata") or {},
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(values: dict) -> dict:
    if values.get("format") != FORMAT:
        raise ValueError("not a Fraise embedding hierarchy")
    if values.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported hierarchy version {values.get('format_version')!r}"
        )
    return values


def _padded(rows, level: int) -> np.ndarray:
    """Accept a padded integer array or ragged lists of children."""
    if isinstance(rows, np.ndarray):
        if rows.ndim != 2:
            raise ValueError(f"level {level} compositions must be two-dimensional")
        return rows.astype(np.int64)
    rows = [list(row) for row in rows]
    width = max(map(len, rows), default=1)
    table = np.full((len(rows), width), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        table[index, : len(row)] = row
    return table


def _short(size: int) -> str:
    for unit, scale in (("M", 1 << 20), ("k", 1 << 10)):
        if size >= scale and size % (scale // 8) == 0:
            return f"{size / scale:g}{unit}"
    return str(size)

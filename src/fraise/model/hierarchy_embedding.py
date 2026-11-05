"""Input embeddings composed bottom-up through a multi-level token hierarchy.

At level ``l >= 1`` every node of the level's vocabulary is represented as

    x_l(n) = compose_l(x_{l-1}(c_1), ..., x_{l-1}(c_k)) + P_l r(n)   if l owns n

where ``c_1 .. c_k`` are ``n``'s ordered children (``[n]`` itself for nodes
carried over from level ``l - 1``) and ``compose_l`` is the order-aware gated
sum of the two-level model. ``x_0`` is the atom table. Each node owns one
residual, added at the level that introduces it (atoms at level 1), so a node
present at several levels never gets duplicate residual parameters. With two
levels this is exactly ``MacroEmbedding``'s compositional formula.

Residual widths vary per node. Values are packed into one flat parameter per
level; node ``n``'s residual is ``values[offset_n : offset_n + width_n]`` and is
projected by the first ``width_n`` rows of the level's projection, so narrow
residuals share the leading directions of wide ones. Width 0 means no residual.

A forward pass deduplicates nodes level by level: each distinct node of the
batch is composed once per level, then gathered back to every position.
Gradients reach every level through those gathers, which use ``F.embedding``
because its backward accumulates much faster than advanced indexing.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fraise.config.hierarchical import FraiseConfig


class GatedSumComposer(nn.Module):
    """``sum_i g_i * x(c_i) / sqrt(k) + length(k)``, plus the residual projection."""

    def __init__(self, max_length: int, dim: int, residual_dim: int):
        super().__init__()
        self.position_gates = nn.Parameter(torch.ones(max_length, dim))
        self.length_embedding = nn.Embedding(max_length + 1, dim)
        nn.init.zeros_(self.length_embedding.weight)
        self.residual_projection = None
        if residual_dim:
            self.residual_projection = nn.Linear(residual_dim, dim, bias=False)
            nn.init.zeros_(self.residual_projection.weight)

    def forward(self, children, lengths, present):
        """``children [nodes, slots, dim]``, ``lengths``/``present`` per slot."""
        terms = children * self.position_gates[: children.size(1)]
        composed = (terms * present.unsqueeze(-1)).sum(1)
        scale = lengths.clamp_min(1).to(composed.dtype).rsqrt().unsqueeze(-1)
        return composed * scale + self.length_embedding(lengths)


class HierarchyLevel(nn.Module):
    """One composed level: child tables for its new nodes and owned residuals."""

    def __init__(self, hierarchy, index: int):
        super().__init__()
        level = hierarchy.levels[index]
        self.index = index
        self.start = hierarchy.levels[index - 1].vocab_size
        self.owned_start = hierarchy.owned_start(index)
        self.residual_dim = level.residual_dim
        new, owned = hierarchy.new_nodes(index), hierarchy.owned_nodes(index)
        width = level.max_composition_length
        # Unbound default: each new node composes atom 0; bind_hierarchy fills them.
        self.register_buffer("compositions", torch.zeros(new, width, dtype=torch.int32))
        self.register_buffer("lengths", torch.ones(new, dtype=torch.int32))
        total = hierarchy.residual_parameters(index)
        self.values = None
        if total:
            self.values = nn.Parameter(torch.randn(total))
        uniform = level.residual_dim if total == owned * level.residual_dim else 0
        self.register_buffer(
            "residual_widths", torch.full((owned,), uniform, dtype=torch.int32)
        )
        self.register_buffer(
            "residual_offsets", torch.arange(owned, dtype=torch.int64) * uniform
        )

    def child_nodes(self, nodes):
        """Ordered children ``[nodes, width]`` and their counts."""
        new = nodes >= self.start
        local = (nodes - self.start).clamp_min(0)
        rows = self.compositions[local].long()
        lengths = self.lengths[local].long()
        carried = torch.zeros_like(rows)
        carried[:, 0] = nodes
        children = torch.where(new.unsqueeze(-1), rows, carried)
        return children, torch.where(new, lengths, 1)

    def residual(self, nodes, projection):
        if self.values is None:
            return None
        owned = nodes >= self.owned_start
        local = (nodes - self.owned_start).clamp_min(0)
        widths = torch.where(owned, self.residual_widths[local].long(), 0)
        slots = torch.arange(self.residual_dim, device=nodes.device)
        index = self.residual_offsets[local].unsqueeze(-1) + slots
        present = slots < widths.unsqueeze(-1)
        index = torch.where(present, index, 0)
        values = F.embedding(index, self.values.unsqueeze(-1)).squeeze(-1) * present
        return F.linear(values, projection.weight[:, : self.residual_dim])


class HierarchicalEmbedding(nn.Module):
    def __init__(self, config: FraiseConfig):
        super().__init__()
        hierarchy = config.embedding_hierarchy
        dim = config.residual_stream_dim
        self.hierarchy = hierarchy
        self.atoms = nn.Embedding(hierarchy.levels[0].vocab_size, dim)
        self.composers = nn.ModuleList(
            GatedSumComposer(length, dim, residual)
            for length, residual in hierarchy.composers()
        )
        self.levels = nn.ModuleList(
            HierarchyLevel(hierarchy, index)
            for index in range(1, len(hierarchy.levels))
        )
        self.register_buffer(
            "input_nodes", torch.arange(config.vocab_size, dtype=torch.int64)
        )

    def composer(self, level: HierarchyLevel) -> GatedSumComposer:
        return self.composers[self.hierarchy.composer_index(level.index)]

    # The number of distinct nodes per level depends on the batch, so this runs
    # eagerly; the transformer after it still compiles as a fixed-shape graph.
    @torch.compiler.disable
    def forward(self, token_ids):
        nodes, inverse = torch.unique(
            self.input_nodes[token_ids].flatten(), return_inverse=True
        )
        plans = []
        for level in reversed(self.levels):
            children, lengths = level.child_nodes(nodes)
            present = torch.arange(children.size(1), device=nodes.device) < (
                lengths.unsqueeze(-1)
            )
            # Padding slots point at node 0 and are masked out of the sum.
            below, index = torch.unique(
                children.masked_fill(~present, 0), return_inverse=True
            )
            plans.append((level, nodes, index, lengths, present))
            nodes = below
        x = self.atoms(nodes)
        for level, nodes, index, lengths, present in reversed(plans):
            composer = self.composer(level)
            x = composer(F.embedding(index, x), lengths, present)
            residual = level.residual(nodes, composer.residual_projection)
            if residual is not None:
                x = x + residual
        return F.embedding(inverse, x).view(*token_ids.shape, -1)

    @torch.no_grad()
    def bind_hierarchy(self, definition) -> None:
        """Copy a checked ``EmbeddingHierarchy`` into the structure buffers."""
        from fraise.hierarchy.binding import check_hierarchy

        check_hierarchy(definition, self.hierarchy, len(self.input_nodes))
        device = self.input_nodes.device
        for level in self.levels:
            rows = torch.as_tensor(definition.compositions[level.index - 1])
            lengths = (rows >= 0).sum(1)
            level.compositions.zero_()
            level.compositions[:, : rows.size(1)] = rows.clamp_min(0).to(device)
            level.lengths.copy_(lengths)
            start, end = definition.owned_range(level.index)
            widths = torch.as_tensor(definition.residual_widths[start:end])
            level.residual_widths.copy_(widths)
            level.residual_offsets.copy_(widths.cumsum(0) - widths)
        nodes = definition.input_node_ids(len(self.input_nodes))
        self.input_nodes.copy_(torch.as_tensor(nodes))

    def definition(self):
        """Reconstruct the bound ``EmbeddingHierarchy`` from the buffers."""
        from fraise.hierarchy.definition import EmbeddingHierarchy

        widths = torch.zeros(self.hierarchy.levels[-1].vocab_size, dtype=torch.int64)
        compositions = []
        for level in self.levels:
            rows = level.compositions.long().cpu()
            slots = torch.arange(rows.size(1))
            present = slots < level.lengths.long().cpu().unsqueeze(-1)
            compositions.append(torch.where(present, rows, -1).numpy())
            start = level.owned_start
            widths[start : start + len(level.residual_widths)] = (
                level.residual_widths.cpu()
            )
        inputs = self.input_nodes.cpu().numpy()
        identity = len(inputs) == len(widths) and bool(
            (inputs == np.arange(len(inputs))).all()
        )
        return EmbeddingHierarchy(
            self.hierarchy.level_sizes,
            compositions,
            widths.numpy(),
            None if identity else inputs,
        )

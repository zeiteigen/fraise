"""Asymmetric Fraise: a global transformer over macro tokens whose next macro is
emitted base token by base token by a small local decoder.

Each macro ID decomposes into at most ``max_macro_length`` base IDs (stored as
persistent buffers bound from the macro tokenizer, so checkpoints are
self-contained). The local decoder predicts, for the macro after position
``t``, its base tokens ``b_1 .. b_k`` followed by an END symbol. Its state for
slot ``j`` is the global context at ``t`` plus an order-aware sum of the base
tokens already emitted for that macro, so every slot is computed independently
and training evaluates only real (non-padding) slots. Generation walks a trie of
macro decompositions, so only complete macros are fed back to the transformer.
"""

import torch
import torch.nn.functional as F
from torch import nn

from fraise.config.hierarchical import FraiseConfig

from .cache import KVCache
from .hierarchy_embedding import HierarchicalEmbedding
from .swiglu import SwiGLU
from .transformer import Transformer


class MacroEmbedding(nn.Module):
    """Compose macro inputs from base embeddings plus a low-rank macro residual.

    ``x(m) = sum_i g_i * E[b_i] / sqrt(k) + length(k) + P r(m)``. Position gates
    ``g_i`` make the composition order-aware. The residual projection starts at
    zero, so every macro begins as its constituents; weight decay pulls rarely
    seen residuals back toward that shared composition.
    """

    def __init__(self, config: FraiseConfig):
        super().__init__()
        dim, length = config.residual_stream_dim, config.max_macro_length
        self.compositional = config.macro_embedding == "compositional"
        if not self.compositional:
            self.table = nn.Embedding(config.vocab_size, dim)
            return
        self.base_embedding = nn.Embedding(config.base_vocab_size, dim)
        self.position_gates = nn.Parameter(torch.ones(length, dim))
        self.length_embedding = nn.Embedding(length + 1, dim)
        nn.init.zeros_(self.length_embedding.weight)
        self.residual = None
        if config.macro_residual_dim:
            self.residual = nn.Embedding(config.vocab_size, config.macro_residual_dim)
            self.residual_projection = nn.Linear(
                config.macro_residual_dim, dim, bias=False
            )
            nn.init.zeros_(self.residual_projection.weight)

    def forward(self, macro_ids, decompositions, lengths):
        if not self.compositional:
            return self.table(macro_ids)
        parts = decompositions[macro_ids]
        counts = lengths[macro_ids]
        offsets = torch.arange(parts.size(-1), device=parts.device)
        present = (offsets < counts.unsqueeze(-1)).unsqueeze(-1)
        terms = self.base_embedding(parts) * self.position_gates
        composed = (terms * present).sum(-2)
        composed = composed * counts.clamp_min(1).to(composed.dtype).rsqrt().unsqueeze(
            -1
        )
        composed = composed + self.length_embedding(counts)
        if self.residual is not None:
            composed = composed + self.residual_projection(self.residual(macro_ids))
        return composed


class LocalBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        return x + self.swiglu(self.norm(x))


class LocalDecoder(nn.Module):
    """Predict one base token (or END) from a context and an emitted prefix."""

    def __init__(self, config: FraiseConfig):
        super().__init__()
        dim, local_dim = config.residual_stream_dim, config.local_dim
        self.end_id = config.base_vocab_size
        self.context_norm = nn.LayerNorm(dim)
        self.context_projection = nn.Linear(dim, local_dim, bias=False)
        self.token_embedding = nn.Embedding(config.base_vocab_size, local_dim)
        self.position_gates = nn.Parameter(
            torch.ones(config.max_macro_length, local_dim)
        )
        self.slot_embedding = nn.Embedding(config.max_macro_length + 1, local_dim)
        nn.init.zeros_(self.slot_embedding.weight)
        self.blocks = nn.ModuleList(
            LocalBlock(local_dim, config.local_swiglu_up_dim)
            for _ in range(config.local_num_layers)
        )
        self.norm = nn.LayerNorm(local_dim)
        self.head = nn.Linear(local_dim, config.base_vocab_size + 1)

    def context(self, hidden):
        return self.context_projection(self.context_norm(hidden))

    def prefix_term(self, base_ids, slots):
        """Contribution of base token(s) emitted at ``slots`` to later states."""
        return self.token_embedding(base_ids) * self.position_gates[slots]

    def logits(self, states, slots):
        """``states`` = context + prefix sum; logits over base IDs and END."""
        x = states + self.slot_embedding(slots)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class FraiseHierarchicalModel(nn.Module):
    hierarchical = True

    def __init__(self, config: FraiseConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.base_vocab_size = config.base_vocab_size
        self.max_macro_length = config.max_macro_length
        self.end_id = config.base_vocab_size
        # Base tokens are their own one-token macros; bind_tokenizer fills the rest.
        decompositions = torch.zeros(
            config.vocab_size, config.max_macro_length, dtype=torch.long
        )
        decompositions[: config.base_vocab_size, 0] = torch.arange(
            config.base_vocab_size
        )
        lengths = torch.zeros(config.vocab_size, dtype=torch.long)
        lengths[: config.base_vocab_size] = 1
        self.register_buffer("decompositions", decompositions)
        self.register_buffer("macro_lengths", lengths)
        self.embedding = (
            HierarchicalEmbedding(config)
            if config.macro_embedding == "hierarchical"
            else MacroEmbedding(config)
        )
        self.transformer = Transformer(
            config.residual_stream_dim,
            config.head_dim,
            config.qk_dim,
            config.num_heads,
            config.swiglu_up_dim,
            config.num_layers,
        )
        self.local = LocalDecoder(config)

    def new_cache(self) -> KVCache:
        return KVCache(len(self.transformer.layers))

    @torch.no_grad()
    def bind_decompositions(self, table, lengths) -> None:
        """Install a tokenizer's ``[vocab, width]`` decompositions and lengths."""
        table = torch.as_tensor(table, dtype=torch.long)
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        vocab, width = table.shape
        if vocab != self.decompositions.size(0) or lengths.shape != (vocab,):
            raise ValueError("tokenizer/model macro vocabulary mismatch")
        if width > self.max_macro_length or int(lengths.max()) > self.max_macro_length:
            raise ValueError("tokenizer macros exceed the model's max_macro_length")
        if int(lengths.min()) < 1 or int(table.max()) >= self.base_vocab_size:
            raise ValueError(
                "tokenizer decompositions fall outside the base vocabulary"
            )
        self.decompositions.zero_()
        self.decompositions[:, :width] = table.to(self.decompositions.device)
        self.macro_lengths.copy_(lengths.to(self.macro_lengths.device))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """Global hidden states ``[batch, sequence, model_dim]`` for macro inputs.

        Hierarchical models do not produce macro-vocabulary logits; training
        uses ``target_loss_sum`` and generation uses the local decoder.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        cached = 0 if cache is None else len(cache)
        if cached + input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len {self.max_seq_len}")
        x = self.embed(input_ids)
        hidden = self.transformer(x, cache)
        return hidden[:, -1:] if last_only else hidden

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Input vectors ``[batch, sequence, model_dim]`` for macro IDs."""
        if isinstance(self.embedding, HierarchicalEmbedding):
            return self.embedding(input_ids)
        return self.embedding(input_ids, self.decompositions, self.macro_lengths)

    def local_targets(self, targets):
        """Per-slot base targets (END after the last part) and their counts."""
        parts = self.decompositions[targets]
        counts = self.macro_lengths[targets]
        slots = torch.arange(self.max_macro_length + 1, device=targets.device)
        slot_targets = torch.where(
            slots < counts.unsqueeze(-1), F.pad(parts, (0, 1)), self.end_id
        )
        return parts, counts, slot_targets

    def target_loss_sum(self, input_ids, targets, target_mask):
        """Summed negative log-likelihood of each valid target macro.

        A macro's NLL is the local decoder's cross-entropy over its base tokens
        and END. Only valid slots reach the decoder and output head.
        """
        hidden = self(input_ids)
        context = self.local.context(hidden)
        parts, counts, slot_targets = self.local_targets(targets)
        width = self.max_macro_length
        offsets = torch.arange(width, device=targets.device)
        present = (offsets < counts.unsqueeze(-1)).unsqueeze(-1)
        terms = self.local.prefix_term(parts, offsets) * present
        # prefix[..., j, :] sums the terms of base tokens before slot j.
        prefix = F.pad(terms.cumsum(-2), (0, 0, 1, 0))
        slots = torch.arange(width + 1, device=targets.device)
        valid = (slots <= counts.unsqueeze(-1)) & target_mask.unsqueeze(-1)
        return _valid_slot_loss(self.local, context, prefix, valid, slot_targets)


# The number of valid slots depends on the data. Keeping this tail out of the
# fixed-shape compiled graph avoids one recompilation per distinct slot count;
# it is dominated by one large output-head matmul, which gains little from it.
@torch.compiler.disable
def _valid_slot_loss(local, context, prefix, valid, slot_targets):
    slots_per_position = valid.size(-1)
    index = valid.flatten().nonzero().squeeze(1)
    positions = torch.div(index, slots_per_position, rounding_mode="floor")
    states = context.flatten(0, 1)[positions] + prefix.flatten(0, 2)[index]
    logits = local.logits(states, index % slots_per_position)
    return F.cross_entropy(
        logits.float(), slot_targets.flatten()[index], reduction="sum"
    )

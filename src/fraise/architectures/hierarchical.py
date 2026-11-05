"""Register the asymmetric Fraise model (macro input, base-token output)."""

from fraise.config.hierarchical import FraiseConfig
from fraise.hierarchy.binding import (
    check_hierarchy,
    check_tokenizer_hierarchy,
    resolve_hierarchy,
)
from fraise.model.hierarchical import FraiseHierarchicalModel
from fraise.model.hierarchy_embedding import HierarchicalEmbedding
from fraise.model.mha import MultiHeadAttention

from .contracts import ArchitectureSpec, OptimizerLayout


def _transformer_layer(config: FraiseConfig) -> int:
    dim, hidden = config.residual_stream_dim, config.swiglu_up_dim
    attention = 2 * dim * config.num_heads * (config.qk_dim + config.head_dim)
    swiglu = 3 * dim * hidden + 2 * hidden + dim
    return attention + swiglu + 4 * dim


def parameter_breakdown(config: FraiseConfig) -> dict[str, int]:
    """Dense transformer weights separately from lexical tables and the head.

    ``macro_lexical`` holds the parameters indexed per macro ID (sparse rows):
    residuals and their projection, or the whole independent embedding table.
    Hierarchical embeddings instead report their composition networks (gates,
    length embeddings, residual projections) and each level's owned residuals.
    """
    dim, local = config.residual_stream_dim, config.local_dim
    length, base = config.max_macro_length, config.base_vocab_size
    lexical = {}
    if config.macro_embedding == "hierarchical":
        hierarchy = config.embedding_hierarchy
        base_lexical, macro_lexical = base * dim, 0
        lexical["composition"] = sum(
            (2 * width + 1 + rank) * dim for width, rank in hierarchy.composers()
        )
        for index in range(1, len(hierarchy.levels)):
            lexical[f"level_{index}_residuals"] = hierarchy.residual_parameters(index)
    elif config.macro_embedding == "independent":
        base_lexical, macro_lexical = 0, config.vocab_size * dim
    else:
        base_lexical = base * dim + length * dim + (length + 1) * dim
        rank = config.macro_residual_dim
        macro_lexical = config.vocab_size * rank + rank * dim
    local_block = 2 * local + 3 * local * config.local_swiglu_up_dim
    local_block += 2 * config.local_swiglu_up_dim + local
    groups = {
        "transformer": config.num_layers * _transformer_layer(config),
        "local_decoder": 2 * dim
        + dim * local
        + length * local
        + (length + 1) * local
        + config.local_num_layers * local_block
        + 2 * local,
        "base_lexical": base_lexical + base * local,
    }
    if config.macro_embedding == "hierarchical":
        groups |= lexical
    else:
        groups["macro_lexical"] = macro_lexical
    groups["output_head"] = local * (base + 1) + base + 1
    return groups


# fp32 value, gradient, and AdamW's two moments.
TRAINING_BYTES_PER_PARAMETER = 16


def hierarchy_report(config: FraiseConfig) -> list[dict]:
    """Per-level residual parameters, their memory, and structure buffers."""
    hierarchy = config.embedding_hierarchy
    records = []
    for index, level in enumerate(hierarchy.levels):
        parameters = hierarchy.residual_parameters(index)
        owned = hierarchy.owned_nodes(index)
        structure = 0
        if index:
            # int32 children and lengths per new node; int32 widths, int64 offsets.
            new = hierarchy.new_nodes(index)
            structure = new * 4 * (level.max_composition_length + 1) + owned * 12
        records.append(
            {
                "level": index,
                "vocab_size": level.vocab_size,
                "new_nodes": hierarchy.new_nodes(index) if index else level.vocab_size,
                "owned_residual_nodes": owned,
                "residual_dim": level.residual_dim,
                "residual_parameters": parameters,
                "residual_bytes": 4 * parameters,
                "residual_training_bytes": TRAINING_BYTES_PER_PARAMETER * parameters,
                "structure_bytes": structure,
            }
        )
    return records


def count_parameters(config: FraiseConfig) -> int:
    return sum(parameter_breakdown(config).values())


def initialize(model, config, settings) -> None:
    """Initialization happens in the modules; zero residual projections included."""


def optimizer_layout(model: FraiseHierarchicalModel) -> OptimizerLayout:
    hidden = [*model.transformer.parameters(), model.local.context_projection.weight]
    for block in model.local.blocks:
        hidden.extend(block.swiglu.parameters())
    residuals = []
    if isinstance(model.embedding, HierarchicalEmbedding):
        residuals = [
            level.values for level in model.embedding.levels if level.values is not None
        ]
    return OptimizerLayout(
        muon_parameters=frozenset(p for p in hidden if p.ndim == 2),
        row_blocks={
            module.qkv_projection.weight: (module.q_size, module.k_size, module.v_size)
            for module in model.modules()
            if isinstance(module, MultiHeadAttention)
        },
        # Packed per-token residuals decay like the two-level residual table.
        decayed_vectors=frozenset(residuals),
    )


def check_tokenizer(config: FraiseConfig, tokenizer) -> None:
    """Require macro decompositions over a base vocabulary matching the model."""
    table = getattr(tokenizer, "decompositions", None)
    if table is None:
        if config.vocab_size != config.base_vocab_size:
            raise ValueError(
                "architecture 'fraise' needs a macro tokenizer when vocab_size "
                "differs from base_vocab_size"
            )
        return
    if tokenizer.base_vocab_size != config.base_vocab_size:
        raise ValueError(
            "macro tokenizer's base vocabulary differs from base_vocab_size"
        )
    if tokenizer.max_macro_length > config.max_macro_length:
        raise ValueError(
            f"macro tokenizer has macros of {tokenizer.max_macro_length} base tokens; "
            f"raise max_macro_length (currently {config.max_macro_length})"
        )


def tokenizer_hierarchy(config: FraiseConfig, tokenizer):
    """The checked hierarchy definition a hierarchical config selects."""
    definition = resolve_hierarchy(config.embedding_hierarchy, tokenizer)
    check_hierarchy(definition, config.embedding_hierarchy, config.vocab_size)
    check_tokenizer_hierarchy(definition, tokenizer)
    return definition


def _check_tokenizer(config: FraiseConfig, tokenizer) -> None:
    check_tokenizer(config, tokenizer)
    if config.embedding_hierarchy is not None:
        tokenizer_hierarchy(config, tokenizer)


def bind_tokenizer(model: FraiseHierarchicalModel, tokenizer) -> None:
    check_tokenizer(model.config, tokenizer)
    table = getattr(tokenizer, "decompositions", None)
    if table is not None:
        model.bind_decompositions(table, tokenizer.unit_lengths)
    if model.config.embedding_hierarchy is not None:
        model.embedding.bind_hierarchy(tokenizer_hierarchy(model.config, tokenizer))


FRAISE = ArchitectureSpec(
    id="fraise",
    config_type=FraiseConfig,
    build=FraiseHierarchicalModel,
    count_parameters=count_parameters,
    initialize=initialize,
    optimizer_layout=optimizer_layout,
    label="Fraise asymmetric (macro in, base out)",
    example_config={
        "architecture": "fraise",
        "vocab_size": 65536,
        "base_vocab_size": 8192,
        "max_macro_length": 8,
        "residual_stream_dim": 512,
        "head_dim": 64,
        "qk_dim": 64,
        "num_heads": 8,
        "swiglu_up_dim": 1536,
        "num_layers": 8,
        "max_seq_len": 512,
        "macro_embedding": "compositional",
        "macro_residual_dim": 32,
        "local_dim": 512,
        "local_num_layers": 2,
        "local_swiglu_up_dim": 1024,
    },
    check_tokenizer=_check_tokenizer,
    bind_tokenizer=bind_tokenizer,
    parameter_breakdown=parameter_breakdown,
)

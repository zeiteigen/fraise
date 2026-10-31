"""Adapt the original Fraise transformer without changing its state-dict keys."""

import torch

from fraise.config.model import ModelConfig
from fraise.model.mha import MultiHeadAttention
from fraise.model.model import FraiseModel

from .contracts import ArchitectureSpec, OptimizerLayout


def count_parameters(config: ModelConfig) -> int:
    vocab_size = config.vocab_size
    model_dim = config.residual_stream_dim
    hidden_dim = config.swiglu_up_dim
    # Tying shares embedding weights; the output projection keeps its own bias.
    weight_matrices = 1 if config.tie_word_embeddings else 2
    embeddings_and_head = weight_matrices * vocab_size * model_dim + vocab_size
    # Attention projections have no biases; SwiGLU and layer norms do.
    attention = 2 * model_dim * config.num_heads * (config.qk_dim + config.head_dim)
    swiglu = 3 * model_dim * hidden_dim + 2 * hidden_dim + model_dim
    layer_norms = 4 * model_dim
    return embeddings_and_head + config.num_layers * (attention + swiglu + layer_norms)


def initialize(model: FraiseModel, config: ModelConfig, settings) -> None:
    if config.tie_word_embeddings:
        # Unit-scale embeddings make the residual stream predict its own token.
        torch.nn.init.normal_(
            model.token_embedding.weight, std=settings.tied_weight_init_std
        )


def optimizer_layout(model: FraiseModel) -> OptimizerLayout:
    return OptimizerLayout(
        muon_parameters=frozenset(
            parameter
            for parameter in model.transformer.parameters()
            if parameter.ndim == 2
        ),
        row_blocks={
            module.qkv_projection.weight: (module.q_size, module.k_size, module.v_size)
            for module in model.modules()
            if isinstance(module, MultiHeadAttention)
        },
    )


TRANSFORMER = ArchitectureSpec(
    id="transformer",
    config_type=ModelConfig,
    build=FraiseModel,
    count_parameters=count_parameters,
    initialize=initialize,
    optimizer_layout=optimizer_layout,
    label="Fraise Transformer",
    example_config={
        "architecture": "transformer",
        "vocab_size": 8192,
        "residual_stream_dim": 512,
        "head_dim": 64,
        "qk_dim": 64,
        "num_heads": 12,
        "swiglu_up_dim": 1024,
        "num_layers": 12,
        "max_seq_len": 512,
        "tie_word_embeddings": True,
    },
)

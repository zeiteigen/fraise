"""Recover the hierarchy a saved run was trained with, from its own weights."""

from pathlib import Path

import torch

from fraise.architectures import model_config_from_dict
from fraise.io import read_json
from fraise.tokenizer.loading import load_tokenizer
from fraise.training.run.artifacts import META_FILE_NAME, resolve_checkpoint
from fraise.training.run.checkpoints import load_model_weights

from .construction import two_level_hierarchy
from .definition import EmbeddingHierarchy


def run_hierarchy(directory: str | Path, checkpoint: str | Path | None = None):
    """``(tokenizer, definition or None, config)`` for a saved run.

    Hierarchical runs read the bound structure buffers from the checkpoint, so
    the result is exactly what training used. Two-level compositional runs are
    presented as their 8k -> 64k hierarchy.
    """
    directory = Path(directory).expanduser()
    metadata = read_json(directory / META_FILE_NAME)
    identity = metadata["manifest"].get("identity", metadata["manifest"])
    config = model_config_from_dict(identity["model_config"])
    tokenizer = load_tokenizer(directory / metadata["tokenizer_file"])
    if config.architecture != "fraise" or config.macro_embedding == "independent":
        return tokenizer, None, config
    if config.macro_embedding == "compositional":
        if getattr(tokenizer, "decompositions", None) is None:
            return tokenizer, None, config
        definition = two_level_hierarchy(
            tokenizer.decompositions,
            tokenizer.unit_lengths,
            config.base_vocab_size,
            config.macro_residual_dim,
        )
        return tokenizer, definition, config
    path = Path(checkpoint) if checkpoint is not None else None
    if path is None:
        final = directory / f"{metadata['model_name']}.pt"
        path = final if metadata.get("training_finished") else resolve_checkpoint(directory)
    elif not path.is_absolute():
        path = directory / path
    return tokenizer, definition_from_weights(config, load_model_weights(path)), config


def definition_from_weights(config, weights: dict) -> EmbeddingHierarchy:
    from fraise.model.hierarchy_embedding import HierarchicalEmbedding

    # Only the integer structure buffers are needed; skip allocating weights.
    with torch.device("meta"):
        embedding = HierarchicalEmbedding(config)
    prefix = "embedding."
    buffers = {
        name: tensor
        for name, tensor in weights.items()
        if name.startswith(prefix) and not tensor.is_floating_point()
    }
    embedding.to_empty(device="cpu")
    with torch.no_grad():
        for name, buffer in embedding.named_buffers():
            buffer.copy_(buffers[prefix + name])
    return embedding.definition()

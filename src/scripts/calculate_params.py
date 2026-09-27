"""Estimate the number of trainable parameters in a Fraise model."""

import argparse
from pathlib import Path

from components.model import ModelConfig
from config import load_model_config


def estimate_parameter_count(config: ModelConfig) -> int:
    """Count model weights and biases from their configured dimensions."""
    vocab_size = config["vocab_size"]
    model_dim = config["residual_stream_dim"]
    head_dim = config["head_dim"]
    qk_dim = config["qk_dim"]
    num_heads = config["num_heads"]
    hidden_dim = config["swiglu_up_dim"]

    # The token embedding and language-model head have separate weights.
    embeddings_and_head = 2 * vocab_size * model_dim + vocab_size

    # Q, K, V, and output projections have no biases.
    attention = 2 * model_dim * num_heads * (qk_dim + head_dim)

    # SwiGLU has three biased linear layers; two layer norms each have a
    # learned weight and bias. RoPE and max_seq_len add no parameters.
    swiglu = 3 * model_dim * hidden_dim + 2 * hidden_dim + model_dim
    layer_norms = 4 * model_dim
    return embeddings_and_head + config["num_layers"] * (
        attention + swiglu + layer_norms
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the number of Fraise model parameters"
    )
    parser.add_argument("model_config", type=Path, help="path to model config JSON")
    args = parser.parse_args(argv)

    try:
        config = load_model_config(args.model_config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(estimate_parameter_count(config))


if __name__ == "__main__":
    main()

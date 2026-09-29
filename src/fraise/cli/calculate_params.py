"""Estimate the number of trainable parameters in a Fraise model."""

import argparse
from pathlib import Path

from fraise.config import load_model_config
from fraise.model.parameters import estimate_parameter_count

from .common import cli_errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the number of Fraise model parameters"
    )
    parser.add_argument("model_config", type=Path, help="path to model config JSON")
    args = parser.parse_args(argv)

    with cli_errors(parser):
        config = load_model_config(args.model_config)
    print(estimate_parameter_count(config))


if __name__ == "__main__":
    main()

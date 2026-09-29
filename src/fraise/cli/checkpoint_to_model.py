"""Extract a standalone model state_dict from a training checkpoint."""

import argparse
from pathlib import Path

from fraise.training.checkpoints import checkpoint_to_model

from .common import cli_errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="training checkpoint .pt file")
    parser.add_argument("output", type=Path, help="standalone model .pt file")
    args = parser.parse_args(argv)

    with cli_errors(parser):
        output = checkpoint_to_model(
            args.checkpoint.expanduser(), args.output.expanduser()
        )
    print(output)


if __name__ == "__main__":
    main()

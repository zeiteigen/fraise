"""Extract a standalone model state_dict from a training checkpoint."""

import argparse
from pathlib import Path

from train.save_mgr import load_checkpoint_file, save_model_state_dict


def checkpoint_to_model(checkpoint_path: Path, model_path: Path) -> Path:
    """Write checkpoint weights in the same format as SaveManager.finish()."""
    if checkpoint_path.resolve() == model_path.resolve():
        raise ValueError("model path must differ from checkpoint path")

    checkpoint = load_checkpoint_file(checkpoint_path)
    return save_model_state_dict(checkpoint["model_state_dict"], model_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="training checkpoint .pt file")
    parser.add_argument("output", type=Path, help="standalone model .pt file")
    args = parser.parse_args(argv)

    try:
        output = checkpoint_to_model(args.checkpoint, args.output)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(output)


if __name__ == "__main__":
    main()

"""Train or resume the run a Train panel plan describes (the MLproject `launch` entrypoint)."""

import argparse
from pathlib import Path

from fraise.io import read_json

from . import train


def train_arguments(directory):
    """The ``train`` command line for the plan in ``directory``."""
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    argv = [
        "--model-config",
        str(directory / "model.json"),
        "--training-hyperparameters",
        str(directory / "training.json"),
        "--tokenizer-file",
        plan["tokenizer_file"],
        "--model-name",
        plan["run_name"],
        "--data-dir",
        plan["data_dir"],
        "--save-path",
        plan["save_path"],
        "--device",
        plan["device"],
        "--ablationkit-metrics",
    ]
    if plan.get("resume"):
        argv += ["--resume", plan["resume"]]
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="a plan directory the Train panel wrote")
    args = parser.parse_args(argv)
    if not (args.plan / "plan.json").is_file():
        parser.error(f"{args.plan} has no plan.json; prepare a run in the Train panel")
    train.main(train_arguments(args.plan))


if __name__ == "__main__":
    main()

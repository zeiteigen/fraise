import argparse
from pathlib import Path

from config import (
    get_dataset_loader,
    load_model_config,
    load_training_hyperparameters,
)
from tokenizer import load_tokenizer
from train import pretrain_transformer


def _get_config_path(path: Path | None, prompt: str) -> Path:
    if path is not None:
        return path.expanduser()

    entered_path = input(prompt).strip()
    if not entered_path:
        raise ValueError("a JSON configuration path is required")
    return Path(entered_path).expanduser()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Fraise language model")
    parser.add_argument(
        "--training-hyperparameters",
        "--hyperparameters",
        dest="training_hyperparameters",
        type=Path,
        default=None,
        help="JSON file containing training hyperparameters",
    )
    parser.add_argument(
        "--model-config",
        dest="model_config",
        type=Path,
        default=None,
        help="JSON file containing model configuration",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint file or run directory to resume",
    )
    parser.add_argument(
        "--save-path",
        default="./data/models",
        help="directory containing model run directories",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="run directory and final model name",
    )
    args = parser.parse_args(argv)

    if not args.model_name:
        args.model_name = input("Enter a name for the model: ").strip()
        if not args.model_name:
            raise ValueError("a model name is required")

    training_hyperparameters_path = _get_config_path(
        args.training_hyperparameters,
        "Path to training hyperparameters JSON: ",
    )
    model_config_path = _get_config_path(
        args.model_config,
        "Path to model configuration JSON: ",
    )
    training_hyperparameters = load_training_hyperparameters(
        training_hyperparameters_path
    )
    model_config = load_model_config(model_config_path)
    dataset = get_dataset_loader()

    # Load the tokenizer
    tokenizer = load_tokenizer(
        vocab_size=model_config["vocab_size"],
        dataset=dataset,
    )

    model_config = dict(model_config)
    model_config["vocab_size"] = tokenizer.get_vocab_size()

    # Pretrain the transformer model
    pretrain_transformer(
        tokenizer=tokenizer,
        dataset=dataset,
        hyperparameters=training_hyperparameters,
        model_config=model_config,
        save_path=args.save_path,
        model_name=args.model_name,
        resume_from=str(args.resume) if args.resume is not None else None,
        config_paths=(training_hyperparameters_path, model_config_path),
    )

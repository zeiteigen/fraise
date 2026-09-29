import argparse
from pathlib import Path

from config import (
    DATASET_LOADERS,
    get_dataset_loader,
    load_model_config,
    load_training_hyperparameters,
)
from tokenizer import load_tokenizer
from train import pretrain_transformer
from train.hyperparameters import _validate_hyperparameters
from train.save_mgr import read_run_model_name


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
        "--tokenizer-file",
        "--tokenizer",
        dest="tokenizer_file",
        type=Path,
        default=None,
        help="tokenizer JSON file to use for this run",
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

    if args.resume is not None and not args.model_name:
        args.model_name = read_run_model_name(args.resume)
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
    tokenizer_path = args.tokenizer_file
    if tokenizer_path is None:
        entered_path = input("Path to tokenizer JSON: ").strip()
        if not entered_path:
            raise ValueError("a tokenizer file path is required")
        tokenizer_path = Path(entered_path).expanduser()

    training_hyperparameters = load_training_hyperparameters(
        training_hyperparameters_path
    )
    model_config = load_model_config(model_config_path)
    _validate_hyperparameters(
        training_hyperparameters,
        model_config,
        DATASET_LOADERS.keys(),
    )
    active_sources = {
        name
        for name, count in training_hyperparameters["dataset_token_counts"].items()
        if count > 0
    }
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() != model_config["vocab_size"]:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.get_vocab_size()} != {model_config['vocab_size']}"
        )

    model_config = dict(model_config)
    model_config["vocab_size"] = tokenizer.get_vocab_size()
    dataset = get_dataset_loader(
        active_sources,
        tokenizer=tokenizer,
        dataset_token_counts=training_hyperparameters["dataset_token_counts"],
    )

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

import argparse
from pathlib import Path

from config import DATASET_LOADER, TRAINING_HYPERPARAMETERS, MODEL_CONFIG
from tokenizer import load_tokenizer
from train import pretrain_transformer
from components import FraiseModel


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the Fraise language model")
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
        default="fraise_model",
        help="run directory and final model name",
    )
    args = parser.parse_args(argv)

    # Load the tokenizer
    tokenizer = load_tokenizer(
        vocab_size=MODEL_CONFIG["vocab_size"],
        dataset=DATASET_LOADER,
    )

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.get_vocab_size()

    # Pretrain the transformer model
    pretrain_transformer(
        transformer_model_class=FraiseModel,
        tokenizer=tokenizer,
        dataset=DATASET_LOADER,
        hyperparameters=TRAINING_HYPERPARAMETERS,
        model_config=model_config,
        save_path=args.save_path,
        model_name=args.model_name,
        resume_from=str(args.resume) if args.resume is not None else None,
    )

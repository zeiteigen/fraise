"""Write a preview of the exact packed training order."""

import argparse
from dataclasses import replace
from pathlib import Path

from fraise.config import load_model_config, load_training_settings
from fraise.data.loading import load_training_data
from fraise.data.sources import SOURCE_NAMES
from fraise.preview.examples import collect_examples
from fraise.preview.html import render_html
from fraise.tokenizer.loading import load_tokenizer

from .common import add_data_dir, cli_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-hyperparameters", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--first-source-only", action="store_true")
    add_data_dir(parser)
    args = parser.parse_args(argv)
    with cli_errors(parser):
        if args.count < 1:
            raise ValueError("--count must be positive")
        settings = load_training_settings(args.training_hyperparameters)
        config = load_model_config(args.model_config)
        settings.validate_sources(SOURCE_NAMES)
        if settings.sequence_length > config.max_seq_len:
            raise ValueError("sequence_length cannot exceed model max_seq_len")
        tokenizer = load_tokenizer(args.tokenizer_file)
        if tokenizer.get_vocab_size() != config.vocab_size:
            raise ValueError("tokenizer/model vocabulary mismatch")
        selected = None
        if args.first_source_only:
            first = next(
                name for name in SOURCE_NAMES if settings.dataset_token_counts[name]
            )
            budget = settings.dataset_token_counts[first]
            if budget < (args.count - 1) * settings.sequence_length + 1:
                raise ValueError(
                    "first source cannot supply --count windows; omit --first-source-only"
                )
            selected = {first}
            settings = replace(
                settings, dataset_token_counts={first: budget}, max_train_tokens=budget
            )
        data = load_training_data(
            settings.dataset_token_counts,
            tokenizer,
            root=args.data_dir,
            validation=False,
            sources=selected,
        )
        output = (args.output or args.data_dir / "training_examples.html").expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_html(
                collect_examples(
                    data,
                    tokenizer,
                    settings,
                    args.count,
                    first_source_only=args.first_source_only,
                ),
                settings,
            )
        )
        print(output.resolve())


if __name__ == "__main__":
    main()

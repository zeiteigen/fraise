"""Rebuild legacy run identity from its archived settings and current source cache."""

import argparse
from pathlib import Path

from fraise.config.training import TrainingSettings
from fraise.data.loading import load_training_data
from fraise.io import read_json
from fraise.tokenizer.loading import load_tokenizer
from fraise.training.migration import migrate_run

from .common import add_data_dir, cli_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    add_data_dir(parser)
    args = parser.parse_args(argv)
    with cli_errors(parser):
        directory = args.run.expanduser()
        metadata = read_json(directory / "model_meta.json")
        if metadata["manifest"].get("manifest_version") == 2:
            print("Run already uses the current manifest")
            return
        settings = TrainingSettings.from_dict(metadata["manifest"]["training"])
        tokenizer = load_tokenizer(directory / metadata["tokenizer_file"])
        data = load_training_data(
            settings.dataset_token_counts, tokenizer, root=args.data_dir
        )
        migrate_run(directory, data)
        print(f"Migrated {directory}; original metadata saved as model_meta.v2.json")


if __name__ == "__main__":
    main()

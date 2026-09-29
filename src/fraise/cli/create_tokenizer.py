"""Create a tokenizer from explicitly selected text sources."""

import argparse

from fraise.config.values import integer, number
from fraise.data.loading import load_tokenizer_sources
from fraise.data.sources import SOURCE_NAMES
from fraise.tokenizer.artifacts import create_tokenizer
from fraise.tokenizer.migration import migrate_metadata

from .common import add_data_dir, cli_errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES)
    parser.add_argument(
        "--source-size-gb",
        type=float,
        default=0.1,
        help="download cap per streaming source",
    )
    parser.add_argument(
        "--max-bytes-per-source",
        type=int,
        help="text bytes used per source; complete documents are retained",
    )
    parser.add_argument("--migrate-metadata", action="store_true")
    add_data_dir(parser)
    args = parser.parse_args(argv)
    with cli_errors(parser):
        directory = args.data_dir / "tokenizers"
        if args.migrate_metadata:
            print(f"Migrated {migrate_metadata(directory)} tokenizer records")
            return
        if not args.sources:
            parser.error("--sources is required when creating a tokenizer")
        integer("vocab_size", args.vocab_size, 259)
        number("source_size_gb", args.source_size_gb)
        if args.max_bytes_per_source is not None:
            integer("max_bytes_per_source", args.max_bytes_per_source)
        sources = load_tokenizer_sources(
            args.sources, root=args.data_dir, size_gb=args.source_size_gb
        )
        print(
            create_tokenizer(
                sources,
                args.vocab_size,
                directory,
                max_bytes_per_source=args.max_bytes_per_source,
            )
        )


if __name__ == "__main__":
    main()

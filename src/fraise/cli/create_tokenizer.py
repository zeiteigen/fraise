"""Create a base tokenizer, or macro merges over one, from explicit text sources."""

import argparse
from pathlib import Path

from fraise.config.values import integer, number
from fraise.data.loading import load_tokenizer_sources
from fraise.data.sources import SOURCE_NAMES
from fraise.tokenizer.artifacts import create_macro_tokenizer, create_tokenizer
from fraise.tokenizer.loading import load_tokenizer
from fraise.tokenizer.macro_training import DEFAULT_MAX_MACRO_LENGTH

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
    parser.add_argument(
        "--base-tokenizer",
        type=Path,
        help="learn a macro vocabulary over this base tokenizer's IDs",
    )
    parser.add_argument(
        "--max-macro-length",
        type=int,
        default=DEFAULT_MAX_MACRO_LENGTH,
        help="maximum base tokens per macro (with --base-tokenizer)",
    )
    parser.add_argument(
        "--segmentation",
        choices=("bpe", "greedy"),
        default="bpe",
        help="macro segmentation: BPE merge replay, or greedy longest match over "
        "a prefix-closed vocabulary (with --base-tokenizer)",
    )
    parser.add_argument(
        "--word-bounded-vocab-size",
        type=int,
        help="bpe: learn this many entries with merges kept inside words, then "
        "merge across words up to --vocab-size (equal to it: word-bounded only)",
    )
    parser.add_argument(
        "--word-bounded",
        action="store_true",
        help="greedy: never extend a macro across a word start",
    )
    parser.add_argument(
        "--whole-word-phrases",
        action="store_true",
        help="bpe with --word-bounded-vocab-size: cross-word merges join whole "
        "words only, so no macro starts or ends inside a word",
    )
    add_data_dir(parser)
    args = parser.parse_args(argv)
    with cli_errors(parser):
        directory = args.data_dir / "tokenizers"
        if not args.sources:
            parser.error("--sources is required when creating a tokenizer")
        integer("vocab_size", args.vocab_size, 259)
        number("source_size_gb", args.source_size_gb)
        if args.max_bytes_per_source is not None:
            integer("max_bytes_per_source", args.max_bytes_per_source)
        base = load_tokenizer(args.base_tokenizer) if args.base_tokenizer else None
        if base is not None:
            if hasattr(base, "base"):
                parser.error(
                    "--base-tokenizer must be a base tokenizer, not a macro tokenizer"
                )
            integer("vocab_size", args.vocab_size, base.get_vocab_size() + 1)
            integer("max_macro_length", args.max_macro_length, 2)
        sources = load_tokenizer_sources(
            args.sources, root=args.data_dir, size_gb=args.source_size_gb
        )
        if base is not None:
            print(
                create_macro_tokenizer(
                    base,
                    sources,
                    args.vocab_size,
                    directory,
                    max_macro_length=args.max_macro_length,
                    max_bytes_per_source=args.max_bytes_per_source,
                    segmentation=args.segmentation,
                    word_bounded_vocab_size=args.word_bounded_vocab_size,
                    word_bounded=args.word_bounded,
                    whole_word_phrases=args.whole_word_phrases,
                )
            )
            return
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

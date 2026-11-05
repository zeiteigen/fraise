"""Build, inspect, and display embedding hierarchies.

``build`` writes a merge-prefix hierarchy from a macro tokenizer and prints the
``embedding_hierarchy`` model configuration that loads it. ``inspect`` reports
levels, residual parameters, and (with a tokenizer) consistency. ``show``
prints how text is segmented and how every input token is composed.
"""

import argparse
import json
from pathlib import Path

from fraise.hierarchy import EmbeddingHierarchy, hierarchy_from_macro_tokenizer
from fraise.hierarchy.binding import check_tokenizer_hierarchy
from fraise.hierarchy.construction import two_level_hierarchy
from fraise.preview.hierarchy import analyze_text, render_text_tree
from fraise.tokenizer.loading import load_tokenizer

from .common import cli_errors


def _sizes(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def _from_arguments(args, tokenizer) -> EmbeddingHierarchy | None:
    if args.hierarchy is not None:
        return EmbeddingHierarchy.load(args.hierarchy)
    if args.level_sizes:
        return hierarchy_from_macro_tokenizer(
            tokenizer, _sizes(args.level_sizes), _sizes(args.residual_dims)
        )
    if getattr(tokenizer, "decompositions", None) is not None:
        return two_level_hierarchy(
            tokenizer.decompositions,
            tokenizer.unit_lengths,
            tokenizer.base_vocab_size,
            _sizes(args.residual_dims)[0],
        )
    return None


def config_snippet(definition: EmbeddingHierarchy, path: Path, sha256: str) -> dict:
    return {
        "macro_embedding": "hierarchical",
        "macro_residual_dim": 0,
        "embedding_hierarchy": {
            "levels": definition.config_levels(),
            "composition": "per_level",
            "source": "file",
            "path": str(path),
            "sha256": sha256,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="write a merge-prefix hierarchy file")
    build.add_argument("--tokenizer-file", type=Path, required=True)
    build.add_argument(
        "--level-sizes", required=True, help="e.g. '8192 16384 65536' (base first)"
    )
    build.add_argument(
        "--residual-dims", default="32", help="one width, or one per composed level"
    )
    build.add_argument("--output", type=Path, required=True, help=".npz or .json")

    inspect = commands.add_parser("inspect", help="summarize a hierarchy file")
    inspect.add_argument("hierarchy", type=Path)
    inspect.add_argument("--tokenizer-file", type=Path)

    show = commands.add_parser("show", help="print segmentation and composition")
    show.add_argument("text")
    show.add_argument("--tokenizer-file", type=Path, required=True)
    show.add_argument("--hierarchy", type=Path, help="hierarchy file")
    show.add_argument("--level-sizes", help="merge-prefix levels instead of a file")
    show.add_argument("--residual-dims", default="32")
    show.add_argument("--special-tokens", action="store_true")
    show.add_argument("--json", action="store_true", help="print the full analysis")

    args = parser.parse_args(argv)
    with cli_errors(parser):
        if args.command == "build":
            tokenizer = load_tokenizer(args.tokenizer_file)
            definition = hierarchy_from_macro_tokenizer(
                tokenizer, _sizes(args.level_sizes), _sizes(args.residual_dims)
            )
            sha256 = definition.save(args.output)
            print(json.dumps(config_snippet(definition, args.output, sha256), indent=2))
        elif args.command == "inspect":
            definition = EmbeddingHierarchy.load(args.hierarchy)
            if args.tokenizer_file is not None:
                check_tokenizer_hierarchy(definition, load_tokenizer(args.tokenizer_file))
                print("Consistent with the tokenizer's decompositions.")
            for record in definition.level_summary():
                print(json.dumps(record))
            print(f"Total residual parameters: {int(definition.residual_widths.sum()):,}")
        else:
            tokenizer = load_tokenizer(args.tokenizer_file)
            definition = _from_arguments(args, tokenizer)
            if definition is not None:
                check_tokenizer_hierarchy(definition, tokenizer)
            analysis = analyze_text(
                args.text,
                tokenizer,
                definition,
                add_special_tokens=args.special_tokens,
            )
            print(
                json.dumps(analysis, indent=1)
                if args.json
                else render_text_tree(analysis)
            )


if __name__ == "__main__":
    main()

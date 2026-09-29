"""Show the first packed training examples without constructing a model."""

import argparse
from bisect import bisect_right
from dataclasses import replace
from html import escape
from pathlib import Path

from config import (
    DATASET_LOADERS,
    get_dataset_loader,
    load_model_config,
    load_training_hyperparameters,
)
from datasets_mgr import get_dataset_sources
from tokenizer import load_tokenizer
from train.budgets import _plan_training_budgets
from train.hyperparameters import TrainingSettings, _validate_hyperparameters
from train.packing import (
    iter_dataset_packed_batches,
    pretokenize_split,
    special_token_id,
)


def collect_examples(
    dataset, tokenizer, settings: TrainingSettings, count: int, *, first_source_only=False
):
    """Read windows through the same tokenization, budget, and batch path as training."""
    sources = get_dataset_sources(dataset, "train")
    train_splits = {
        name: pretokenize_split(
            source,
            tokenizer,
            tokenization_batch_size=settings.tokenization_batch_size,
            max_tokens=settings.dataset_token_counts[name],
        )
        for name, source in sources.items()
        if settings.dataset_token_counts[name] > 0
    }
    epoch_budgets = _plan_training_budgets(train_splits, settings)
    if first_source_only:
        first_budget = next(iter(epoch_budgets[0].values()))
        if first_budget < (count - 1) * settings.sequence_length + 1:
            raise ValueError(
                "the first source's first epoch cannot supply all requested "
                "windows; run without --first-source-only"
            )
    pad_id = special_token_id(tokenizer, "<|pad|>")
    examples = []

    for epoch, budgets in enumerate(epoch_budgets):
        names = []
        ends = []
        for name in train_splits:
            budget = budgets.get(name, 0)
            if budget:
                names.append(name)
                ends.append((ends[-1] if ends else 0) +
                            (budget + settings.sequence_length - 1) // settings.sequence_length)

        for batch_index, batch in iter_dataset_packed_batches(
            train_splits,
            budgets,
            settings.sequence_length,
            settings.packing_stride,
            pad_id,
            batch_size=settings.batch_size,
            start_batch_index=0,
            seed=settings.seed + epoch,
            shuffle=settings.shuffle,
        ):
            for position in range(len(batch.input_ids)):
                block_index = batch_index * settings.batch_size + position
                valid_targets = int(batch.target_mask[position].sum().item())
                input_ids = batch.input_ids[position].tolist()
                target_ids = batch.targets[position].tolist()
                mask = batch.target_mask[position].tolist()
                examples.append({
                    "number": len(examples) + 1,
                    "epoch": epoch + 1,
                    "source": names[bisect_right(ends, block_index)],
                    "batch": batch_index,
                    "batch_position": position,
                    "valid_targets": valid_targets,
                    "input_text": tokenizer.decode(input_ids, skip_special_tokens=False),
                    "target_text": tokenizer.decode(target_ids[:valid_targets], skip_special_tokens=False),
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "target_mask": mask,
                })
                if len(examples) == count:
                    return examples
    raise ValueError(
        f"training produces only {len(examples)} packed examples; requested {count}"
    )


def render_html(examples, settings: TrainingSettings) -> str:
    """Make the packed windows readable while retaining every tensor value."""
    cards = []
    for example in examples:
        source = escape(example["source"])
        fields = []
        for title, key in (
            ("Input text", "input_text"),
            ("Target text", "target_text"),
        ):
            fields.append(f"<h3>{title}</h3><pre>{escape(example[key])}</pre>")
        for title, key in (
            ("Input IDs", "input_ids"),
            ("Target IDs", "target_ids"),
            ("Target mask (1 = loss target)", "target_mask"),
        ):
            values = example[key]
            if key == "target_mask":
                values = [int(value) for value in values]
            fields.append(f"<details><summary>{title}</summary><pre>{escape(str(values))}</pre></details>")
        cards.append(
            f'<article><h2>#{example["number"]} · {source}</h2>'
            f'<p>Epoch {example["epoch"]} · Batch {example["batch"]} · '
            f'Position {example["batch_position"]} · '
            f'{example["valid_targets"]} valid targets</p>'
            + "".join(fields) + "</article>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Fraise training examples</title>"
        "<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;"
        "background:#f7f7f4;color:#222}article{background:white;border:1px solid #ddd;"
        "border-radius:8px;padding:1rem 1.5rem;margin:1.5rem 0}h2{margin:0}h3{margin-bottom:.3rem}"
        "p{color:#555}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f0;"
        "padding:.8rem;border-radius:4px}details{margin:.7rem 0}summary{cursor:pointer}</style>"
        f"<h1>Fraise training examples</h1><p>{len(examples)} packed windows · "
        f"sequence length {settings.sequence_length} · stride {settings.packing_stride} · "
        f"batch size {settings.batch_size} · shuffle {settings.shuffle} · seed {settings.seed}. "
        "Input text includes every model input position; target text shows only loss "
        "positions. Expand the ID and mask "
        "sections for the complete tensors, including padding.</p>"
        + "".join(cards) + "</html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-hyperparameters", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/training_examples.html"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--first-source-only",
        action="store_true",
        help="Load only the first configured source when its first epoch covers all requested windows",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")

    hyperparameters = load_training_hyperparameters(args.training_hyperparameters)
    model_config = load_model_config(args.model_config)
    settings = _validate_hyperparameters(
        hyperparameters, model_config, DATASET_LOADERS.keys()
    )
    tokenizer = load_tokenizer(args.tokenizer_file)
    if tokenizer.get_vocab_size() != model_config["vocab_size"]:
        raise ValueError("tokenizer/model vocabulary mismatch")
    active_sources = {
        name for name, budget in settings.dataset_token_counts.items() if budget > 0
    }
    if args.first_source_only:
        # Training exhausts each source's epoch budget before moving to the
        # next. If this source covers the preview, later sources cannot change
        # any of its windows, even though loading them can be expensive.
        first_source = next(name for name in DATASET_LOADERS if name in active_sources)
        first_budget = settings.dataset_token_counts[first_source]
        if first_budget < (args.count - 1) * settings.sequence_length + 1:
            parser.error("the first source cannot supply --count windows; run without --first-source-only")
        active_sources = {first_source}
        settings = replace(
            settings,
            dataset_token_counts={first_source: first_budget},
            max_train_tokens=first_budget,
        )
    dataset = get_dataset_loader(
        active_sources,
        tokenizer=tokenizer,
        dataset_token_counts=settings.dataset_token_counts,
    )
    examples = collect_examples(
        dataset, tokenizer, settings, args.count,
        first_source_only=args.first_source_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(examples, settings), encoding="utf-8")
    print(f"Wrote {len(examples)} training examples to {args.output.resolve()}")


if __name__ == "__main__":
    main()

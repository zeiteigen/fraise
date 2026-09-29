"""Collect decoded examples from the shared preparation and batch iterators."""

from fraise.config.values import integer
from fraise.tokenizer.loading import special_token_id
from fraise.training.batches import iter_dataset_packed_batches
from fraise.training.preparation import prepare_data


def collect_examples(data, tokenizer, settings, count, *, first_source_only=False):
    integer("count", count)
    prepared = prepare_data(data, tokenizer, settings)
    if first_source_only:
        first_budget = next(iter(prepared.budgets[0].values()))
        if first_budget < (count - 1) * settings.sequence_length + 1:
            raise ValueError(
                "first source's first epoch cannot supply --count windows; omit --first-source-only"
            )
    pad_id = special_token_id(tokenizer, "<|pad|>")
    examples = []
    for epoch, budgets in enumerate(prepared.budgets):
        for batch_index, batch in iter_dataset_packed_batches(
            prepared.train,
            budgets,
            settings.sequence_length,
            settings.packing_stride,
            pad_id,
            batch_size=settings.batch_size,
            seed=settings.seed + epoch,
            shuffle=settings.shuffle,
        ):
            for position, source in enumerate(batch.sources):
                valid_targets = int(batch.target_mask[position].sum().item())
                inputs = batch.input_ids[position].tolist()
                targets = batch.targets[position].tolist()
                examples.append(
                    {
                        "number": len(examples) + 1,
                        "epoch": epoch + 1,
                        "source": source,
                        "batch": batch_index,
                        "batch_position": position,
                        "valid_targets": valid_targets,
                        "input_text": tokenizer.decode(
                            inputs, skip_special_tokens=False
                        ),
                        "target_text": tokenizer.decode(
                            targets[:valid_targets], skip_special_tokens=False
                        ),
                        "input_ids": inputs,
                        "target_ids": targets,
                        "target_mask": batch.target_mask[position].tolist(),
                    }
                )
                if len(examples) == count:
                    return examples
    raise ValueError(
        f"training produces only {len(examples)} packed examples; requested {count}"
    )

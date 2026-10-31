"""Evaluate token-weighted held-out loss with bounded minibatches."""

import math
from time import monotonic

import torch

from ..batch import autocast_context, loss_for_batch
from ..execution import deterministic_execution
from ..pipeline.batches import iter_dataset_packed_batches
from ..pipeline.packing import packed_target_capacity, plan_unit_windows
from .measures import ValidationMeasures


def evaluate(model, split, settings, pad_token_id, *, progress=None):
    """Mean held-out loss per target, or ``None`` when validation is disabled."""
    scores = evaluate_scores(model, split, settings, pad_token_id, progress=progress)
    return None if scores is None else scores["loss"]


def evaluate_scores(
    model, split, settings, pad_token_id, *, progress=None, measures=None
):
    """Loss per target plus, when byte lengths are known, bits per byte.

    ``max_validation_tokens`` counts targets for base tokenizers and base-token
    units for macro tokenizers, so every architecture scores the same text.
    """
    if settings.max_validation_tokens == 0:
        return None
    measures = measures or ValidationMeasures()
    units = measures.unit_lengths
    device = next(model.parameters()).device
    if units is None:
        capacity = packed_target_capacity(
            split, settings.sequence_length, settings.packing_stride
        )
        budget = (
            min(capacity, settings.max_validation_tokens)
            if settings.max_validation_tokens is not None
            else capacity
        )
    else:
        # The exact units of the macros that reach the limit (or all of them).
        budget = plan_unit_windows(
            split,
            settings.sequence_length,
            settings.packing_stride,
            settings.max_validation_tokens,
            units,
        ).units
    total_loss = total_tokens = total_units = total_bytes = 0
    was_training = model.training
    started = last_report = monotonic()
    if progress is not None:
        progress(f"Evaluating {budget:,} validation targets.")
    model.eval()
    try:
        with (
            deterministic_execution(settings.deterministic),
            torch.inference_mode(),
        ):
            for _, batch in iter_dataset_packed_batches(
                {"validation": split},
                {"validation": budget},
                settings.sequence_length,
                settings.packing_stride,
                pad_token_id,
                batch_size=settings.minibatch_size,
                unit_lengths=units,
            ):
                valid_tokens = int(batch.target_mask.sum().item())
                targets = batch.targets[batch.target_mask].numpy()
                with autocast_context(device, settings.use_bf16):
                    loss = loss_for_batch(model, batch.to(device), valid_tokens)
                total_loss += float(loss.item()) * valid_tokens
                total_tokens += valid_tokens
                total_units += (
                    valid_tokens if units is None else int(units[targets].sum())
                )
                if measures.byte_lengths is not None:
                    total_bytes += int(measures.byte_lengths[targets].sum())
                now = monotonic()
                if progress is not None and (
                    now - last_report >= 10 or total_units == budget
                ):
                    elapsed = now - started
                    progress(
                        f"Progress {total_units:,}/{budget:,} targets, "
                        f"{elapsed:.1f}s elapsed, {total_units / max(elapsed, 0.001):,.0f} targets/s."
                    )
                    last_report = now
    finally:
        model.train(was_training)
    if not total_tokens:
        return None
    return {
        "loss": total_loss / total_tokens,
        "bits_per_byte": total_loss / math.log(2) / total_bytes
        if total_bytes
        else None,
        "targets": total_tokens,
        "base_tokens": total_units,
        "bytes": total_bytes or None,
    }

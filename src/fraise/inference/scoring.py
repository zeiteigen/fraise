"""Score text and inspect the next-token distribution."""

import math

import torch

from fraise.tokenizer.loading import special_token_id
from fraise.training.batch import token_loss_sum


def score_text(model, tokenizer, text: str) -> dict:
    ids = [
        special_token_id(tokenizer, "<|bos|>"),
        *tokenizer.encode(text, add_special_tokens=False).ids,
        special_token_id(tokenizer, "<|eos|>"),
    ]
    original_count = len(ids) - 1
    ids = ids[: model.max_seq_len + 1]
    device = next(model.parameters()).device
    inputs = torch.tensor([ids[:-1]], device=device)
    targets = torch.tensor([ids[1:]], device=device)
    with torch.inference_mode():
        total = token_loss_sum(
            model, inputs, targets, torch.ones_like(targets, dtype=torch.bool)
        )
    total = float(total.item())
    value = total / (len(ids) - 1)
    truncated = original_count > len(ids) - 1
    text_bytes = len(text.encode("utf-8"))
    return {
        "tokens_scored": len(ids) - 1,
        "truncated": truncated,
        "loss": value,
        "perplexity": math.exp(min(value, 80.0)),
        # The scored targets include EOS, which covers no text bytes.
        "bits_per_byte": total / math.log(2) / text_bytes
        if text_bytes and not truncated
        else None,
    }


def top_next_tokens(model, tokenizer, text: str, count=10):
    from .generation import logits_for_context, prompt_ids

    if getattr(model, "hierarchical", False):
        from .hierarchical import next_base_distribution

        options, probabilities = next_base_distribution(model, tokenizer, text)
        values, indices = probabilities.topk(min(count, probabilities.numel()))
        return [
            (
                token_id,
                probability,
                tokenizer.decode([token_id], skip_special_tokens=False),
            )
            for token_id, probability in zip(options[indices].tolist(), values.tolist())
        ]
    logits = logits_for_context(model, prompt_ids(tokenizer, text))
    values, ids = logits.softmax(-1).topk(min(count, logits.numel()))
    return [
        (token_id, probability, tokenizer.decode([token_id], skip_special_tokens=False))
        for token_id, probability in zip(ids.tolist(), values.tolist())
    ]

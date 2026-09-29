"""Score text and inspect the next-token distribution."""

import math

import torch
import torch.nn.functional as F

from fraise.tokenizer.loading import special_token_id


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
    targets = torch.tensor(ids[1:], device=device)
    with torch.inference_mode():
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets)
    value = float(loss.item())
    return {
        "tokens_scored": len(ids) - 1,
        "truncated": original_count > len(ids) - 1,
        "loss": value,
        "perplexity": math.exp(min(value, 80.0)),
    }


def top_next_tokens(model, tokenizer, text: str, count=10):
    from .generation import logits_for_context, prompt_ids

    logits = logits_for_context(model, prompt_ids(tokenizer, text))
    values, ids = logits.softmax(-1).topk(min(count, logits.numel()))
    return [
        (token_id, probability, tokenizer.decode([token_id], skip_special_tokens=False))
        for token_id, probability in zip(ids.tolist(), values.tolist())
    ]

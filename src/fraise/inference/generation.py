"""Autoregressive generation with greedy, top-k, or nucleus sampling."""

import torch

from fraise.config.values import integer, number
from fraise.tokenizer.loading import special_token_id


def prompt_ids(tokenizer, text):
    return [
        special_token_id(tokenizer, "<|bos|>"),
        *tokenizer.encode(text, add_special_tokens=False).ids,
    ]


def logits_for_context(model, token_ids):
    device = next(model.parameters()).device
    inputs = torch.tensor(
        [token_ids[-model.max_seq_len :]], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        return model(inputs)[0, -1]


def sample_token(logits, temperature, top_k, top_p):
    if temperature == 0:
        return int(logits.argmax().item())
    logits = logits / temperature
    if top_k is not None:
        threshold = logits.topk(min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None:
        values, indices = logits.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.scatter(0, indices, values.masked_fill(remove, float("-inf")))
    return int(torch.multinomial(logits.softmax(-1), 1).item())


def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens=48,
    temperature=0.0,
    top_k=None,
    top_p=0.95,
):
    integer("max_new_tokens", max_new_tokens, 0)
    number("temperature", temperature, inclusive=True)
    if top_k is not None:
        integer("top_k", top_k)
    if top_p is not None:
        number("top_p", top_p)
        if top_p > 1:
            raise ValueError("top_p must be <= 1")
    ids = prompt_ids(tokenizer, prompt)
    eos = special_token_id(tokenizer, "<|eos|>")
    for _ in range(max_new_tokens):
        token_id = sample_token(
            logits_for_context(model, ids), temperature, top_k, top_p
        )
        ids.append(token_id)
        if token_id == eos:
            break
    return tokenizer.decode(ids[1:], skip_special_tokens=True), ids

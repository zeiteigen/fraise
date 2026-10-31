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
    """Last-position logits for the most recent ``max_seq_len`` tokens."""
    device = next(model.parameters()).device
    inputs = torch.tensor(
        [token_ids[-model.max_seq_len :]], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        return model(inputs, last_only=True)[0, -1]


class IncrementalDecoder:
    """Next-token logits that reuse cached keys/values while the context fits.

    Once the sequence exceeds ``max_seq_len`` the window slides and every
    position changes, so it falls back to recomputing the visible window.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.cache = None

    def logits(self, token_ids) -> torch.Tensor:
        if len(token_ids) > self.model.max_seq_len:
            self.cache = None
            return logits_for_context(self.model, token_ids)
        if self.cache is None:
            self.cache = self.model.new_cache()
        if self.cache is None:
            return logits_for_context(self.model, token_ids)
        new = token_ids[len(self.cache) :]
        inputs = torch.tensor([new], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self.model(inputs, cache=self.cache, last_only=True)[0, -1]


def sample_token(logits, temperature, top_k, top_p, *, generator=None):
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
    return int(torch.multinomial(logits.softmax(-1), 1, generator=generator).item())


def validate_sampling(max_new_tokens, temperature, top_k, top_p):
    integer("max_new_tokens", max_new_tokens, 0)
    number("temperature", temperature, inclusive=True)
    if top_k is not None:
        integer("top_k", top_k)
    if top_p is not None:
        number("top_p", top_p)
        if top_p > 1:
            raise ValueError("top_p must be <= 1")


def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens=48,
    temperature=0.0,
    top_k=None,
    top_p=0.95,
    generator=None,
):
    from .streaming import iter_generate

    for step in iter_generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    ):
        pass
    return step.text, step.ids

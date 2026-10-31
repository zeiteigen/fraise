"""Cancellable token generation with cumulative decoded text snapshots."""

from dataclasses import dataclass

from fraise.tokenizer.loading import special_token_id

from .generation import IncrementalDecoder, prompt_ids, sample_token, validate_sampling


@dataclass
class GenerationStep:
    ids: list[int]
    text: str
    completion: str
    generated_tokens: int
    finish_reason: str | None


def iter_generate(
    model,
    tokenizer,
    prompt,
    *,
    max_new_tokens=48,
    temperature=0.0,
    top_k=None,
    top_p=0.95,
    generator=None,
    cancelled=lambda: False,
):
    """Snapshots replace earlier text, including incomplete byte sequences.

    Hierarchical models generate base tokens; ``ids`` then holds base IDs, which
    are also valid macro IDs for decoding, and token counts are base tokens.
    """
    if getattr(model, "hierarchical", False):
        yield from _iter_generate_hierarchical(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            cancelled=cancelled,
        )
        return
    validate_sampling(max_new_tokens, temperature, top_k, top_p)
    ids = prompt_ids(tokenizer, prompt)
    initial = len(ids)
    eos = special_token_id(tokenizer, "<|eos|>")
    decoder = IncrementalDecoder(model)

    def snapshot(reason=None):
        return GenerationStep(
            ids.copy(),
            tokenizer.decode(ids[1:], skip_special_tokens=True),
            tokenizer.decode(ids[initial:], skip_special_tokens=True),
            len(ids) - initial,
            reason,
        )

    for index in range(max_new_tokens):
        if cancelled():
            yield snapshot("cancelled")
            return
        token = sample_token(
            decoder.logits(ids), temperature, top_k, top_p, generator=generator
        )
        ids.append(token)
        reason = (
            "eos"
            if token == eos
            else ("length" if index + 1 == max_new_tokens else None)
        )
        yield snapshot(reason)
        if reason is not None:
            return
    if max_new_tokens == 0:
        yield snapshot("length")


def _iter_generate_hierarchical(model, tokenizer, prompt, **options):
    from .hierarchical import iter_generate_base_tokens

    for step in iter_generate_base_tokens(model, tokenizer, prompt, **options):
        initial = len(step.base_ids) - step.generated_tokens
        yield GenerationStep(
            step.base_ids,
            tokenizer.decode(step.base_ids[1:], skip_special_tokens=True),
            tokenizer.decode(step.base_ids[initial:], skip_special_tokens=True),
            step.generated_tokens,
            step.finished,
        )

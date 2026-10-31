"""Measure greedy decoding throughput through the generation decoder."""

import time

import torch

from fraise.config.values import integer

from .generation import IncrementalDecoder, prompt_ids


def benchmark_generation(
    model, tokenizer, prompt, *, max_new_tokens=32, warmup_runs=1, timed_runs=3
):
    integer("max_new_tokens", max_new_tokens)
    integer("warmup_runs", warmup_runs, 0)
    integer("timed_runs", timed_runs)
    device = next(model.parameters()).device
    if getattr(model, "hierarchical", False):
        return _benchmark_hierarchical(
            model, tokenizer, prompt, max_new_tokens, warmup_runs, timed_runs, device
        )
    initial = prompt_ids(tokenizer, prompt)
    elapsed = []
    for run in range(warmup_runs + timed_runs):
        ids = initial.copy()
        decoder = IncrementalDecoder(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max_new_tokens):
            ids.append(int(decoder.logits(ids).argmax().item()))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if run >= warmup_runs:
            elapsed.append(time.perf_counter() - start)
    seconds = sum(elapsed)
    return {
        "prompt_tokens": len(initial),
        "tokens_per_run": max_new_tokens,
        "timed_runs": timed_runs,
        "total_seconds": seconds,
        "tokens_per_second": max_new_tokens * timed_runs / seconds,
    }


def _benchmark_hierarchical(
    model, tokenizer, prompt, max_new_tokens, warmup_runs, timed_runs, device
):
    """Greedy base tokens per second, plus global transformer steps per token."""
    from .hierarchical import iter_generate_base_tokens

    elapsed, tokens, global_steps = [], 0, 0
    for run in range(warmup_runs + timed_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for step in iter_generate_base_tokens(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        ):
            pass
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if run >= warmup_runs:
            elapsed.append(time.perf_counter() - start)
            tokens += step.generated_tokens
            global_steps += step.global_steps
    seconds = sum(elapsed)
    return {
        "prompt_tokens": len(prompt_ids(tokenizer, prompt)),
        "tokens_per_run": tokens / timed_runs,
        "timed_runs": timed_runs,
        "total_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "global_steps_per_token": global_steps / max(tokens, 1),
    }

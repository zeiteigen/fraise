"""Measure greedy full-context decoding throughput without a KV cache."""

import time

import torch

from fraise.config.values import integer

from .generation import logits_for_context, prompt_ids


def benchmark_generation(
    model, tokenizer, prompt, *, max_new_tokens=32, warmup_runs=1, timed_runs=3
):
    integer("max_new_tokens", max_new_tokens)
    integer("warmup_runs", warmup_runs, 0)
    integer("timed_runs", timed_runs)
    device = next(model.parameters()).device
    initial = prompt_ids(tokenizer, prompt)
    elapsed = []
    for run in range(warmup_runs + timed_runs):
        ids = initial.copy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max_new_tokens):
            ids.append(int(logits_for_context(model, ids).argmax().item()))
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

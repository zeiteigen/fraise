"""Generate continuations from saved checkpoints, one model at a time on one device."""

from time import perf_counter

import torch

from fraise.inference.artifacts import load_run
from fraise.inference.generation import prompt_ids
from fraise.inference.streaming import iter_generate


class InferenceSession:
    """Keeps the most recently used checkpoint loaded until another one is needed."""

    def __init__(self, device):
        self.device, self.loaded, self.key = device, None, None

    def unload(self):
        self.loaded, self.key = None, None
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def load(self, directory, checkpoint):
        stat = checkpoint.stat()
        metadata = (directory / "model_meta.json").stat()
        key = (
            str(checkpoint),
            stat.st_mtime_ns,
            stat.st_size,
            metadata.st_mtime_ns,
            metadata.st_size,
        )
        if key != self.key:
            self.unload()
            self.loaded = load_run(directory, checkpoint=checkpoint, device=self.device)
            self.key = key
        return self.loaded


def generate_sequentially(request, references, session, on_event, cancelled=lambda: False):
    """Continue ``request.prompt`` with each ``(reference, directory, checkpoint)`` in turn.

    Every model starts from the same seed. ``on_event(kind, payload)`` receives ``loading``,
    ``output`` (after each token) and ``result`` per model; the first failure is reported as
    ``error`` and ends the sequence. Returns the ``result`` payloads.
    """
    results = []
    index = None
    try:
        for index, (reference, directory, checkpoint) in enumerate(references):
            if cancelled():
                break
            base = {"index": index, "run_id": reference.run_id}
            on_event("loading", base)
            started = perf_counter()
            loaded = session.load(directory, checkpoint)
            load_seconds = perf_counter() - started
            generator = torch.Generator(device=session.device).manual_seed(request.seed)
            prompt_tokens = len(prompt_ids(loaded.tokenizer, request.prompt))
            started = perf_counter()
            step = None
            for step in iter_generate(
                loaded.model,
                loaded.tokenizer,
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                generator=generator,
                cancelled=cancelled,
            ):
                on_event(
                    "output",
                    {**base, "text": step.completion, "generated_tokens": step.generated_tokens},
                )
            seconds = perf_counter() - started
            result = {
                **base,
                "checkpoint_id": checkpoint.name,
                "text": step.completion if step else "",
                "generated_tokens": step.generated_tokens if step else 0,
                "seconds": seconds,
                "tokens_per_second": (step.generated_tokens if step else 0) / max(seconds, 1e-9),
                "load_seconds": load_seconds,
                "finish_reason": step.finish_reason if step else "cancelled",
                "prompt_tokens": prompt_tokens,
            }
            results.append(result)
            on_event("result", result)
            # Drop the local reference before the next model is loaded.
            del loaded
    except Exception as error:
        on_event("error", {"index": index, "message": str(error)})
    return results

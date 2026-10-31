"""Stream sequential model completions from one cancellable compute worker."""

import asyncio
import json
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4

import torch

from fraise.inference.artifacts import load_run
from fraise.inference.generation import prompt_ids
from fraise.inference.streaming import iter_generate


class InferenceSession:
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


class GenerationStream:
    def __init__(self, request, references, session, gate):
        self.id = uuid4().hex
        self.request, self.references, self.session, self.gate = (
            request,
            references,
            session,
            gate,
        )
        self.cancelled, self.finished = Event(), Event()
        self.queue = Queue(maxsize=16)
        self.thread = Thread(target=self._produce, daemon=True)
        gate.acquire("generation", self.id)
        try:
            self.thread.start()
        except BaseException:
            gate.release(self.id)
            raise

    def _put(self, kind, payload):
        while not self.cancelled.is_set():
            try:
                self.queue.put((kind, payload), timeout=0.1)
                return
            except Full:
                continue

    def _produce(self):
        index = None
        try:
            prompt = self.request.prompt
            for index, (reference, directory, checkpoint) in enumerate(self.references):
                if self.cancelled.is_set():
                    break
                base = {"index": index, "run_id": reference.run_id}
                self._put("loading", base)
                started = perf_counter()
                loaded = self.session.load(directory, checkpoint)
                load_seconds = perf_counter() - started
                generator = torch.Generator(device=self.session.device).manual_seed(
                    self.request.seed
                )
                prompt_tokens = len(prompt_ids(loaded.tokenizer, prompt))
                started = perf_counter()
                for step in iter_generate(
                    loaded.model,
                    loaded.tokenizer,
                    prompt,
                    max_new_tokens=self.request.max_new_tokens,
                    temperature=self.request.temperature,
                    top_k=self.request.top_k,
                    top_p=self.request.top_p,
                    generator=generator,
                    cancelled=self.cancelled.is_set,
                ):
                    self._put(
                        "output",
                        {
                            **base,
                            "text": step.completion,
                            "generated_tokens": step.generated_tokens,
                        },
                    )
                seconds = perf_counter() - started
                self._put(
                    "result",
                    {
                        **base,
                        "text": step.completion,
                        "generated_tokens": step.generated_tokens,
                        "seconds": seconds,
                        "tokens_per_second": step.generated_tokens / max(seconds, 1e-9),
                        "load_seconds": load_seconds,
                        "finish_reason": step.finish_reason,
                        "prompt_tokens": prompt_tokens,
                    },
                )
                # Drop the local reference before the next model is loaded.
                del loaded
            self._put("done", {})
        except Exception as error:
            self._put("error", {"index": index, "message": str(error)})
            self._put("done", {})
        finally:
            self.gate.release(self.id)
            self.finished.set()

    async def events(self):
        try:
            while not self.finished.is_set() or not self.queue.empty():
                try:
                    kind, payload = await asyncio.to_thread(self.queue.get, True, 0.1)
                except Empty:
                    continue
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            self.cancelled.set()
            # The gate stays occupied until the worker has actually left its token step.
            await asyncio.to_thread(self.thread.join)

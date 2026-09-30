"""Load checkpoints and evaluate held-out targets in an isolated process."""

import os
from dataclasses import dataclass, replace
from time import monotonic

import torch
from datasets import Dataset

from fraise.config.training import TrainingSettings
from fraise.model.model import FraiseModel

from .checkpoints import load_checkpoint_file
from .validation import evaluate
from .worker_lifetime import exit_with_parent


@dataclass
class CheckpointValidator:
    model: FraiseModel
    split: Dataset
    settings: TrainingSettings
    pad_token_id: int

    def evaluate_checkpoint(self, path, *, progress=None):
        if progress is not None:
            progress("Loading checkpoint weights.")
        checkpoint = load_checkpoint_file(path, mmap=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        model = self.model
        settings = self.settings
        if settings.validation_device == "npu":
            from .validation_npu import OpenVinoNpuModel

            model = OpenVinoNpuModel(model, settings.sequence_length, progress=progress)
            settings = replace(settings, minibatch_size=1)
        return evaluate(
            model, self.split, settings, self.pad_token_id, progress=progress
        )


_worker: CheckpointValidator | None = None
_events = None


def init_validation_worker(config, split, settings, pad_token_id, parent_pid, events):
    global _worker, _events
    exit_with_parent(parent_pid)
    _events = events
    torch.set_num_threads(settings.validation_cpu_threads)
    _worker = CheckpointValidator(
        FraiseModel(config).to("cpu").eval(),
        split,
        replace(settings, use_bf16=False),
        pad_token_id,
    )
    events.send(
        f"Validation worker ready: PID {os.getpid()}, backend {settings.validation_device}, "
        f"CPU preparation/loss: {settings.validation_cpu_threads} threads, FP32."
    )


def evaluate_checkpoint(step, path):
    if _worker is None:
        raise RuntimeError("validation worker was not initialized")
    started = monotonic()

    def progress(message):
        _events.send(f"Validation step {step}: {message}")

    progress(f"Started checkpoint {path}.")
    loss = _worker.evaluate_checkpoint(path, progress=progress)
    progress(
        f"Worker finished in {monotonic() - started:.1f}s; loss {loss}; result ready for training."
    )
    return loss

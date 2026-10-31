"""Load checkpoints and evaluate held-out targets in an isolated process."""

import os
from dataclasses import dataclass, replace
from time import monotonic

import torch
from datasets import Dataset

from fraise.architectures import ModelProtocol, get_architecture
from fraise.config.training import TrainingSettings

from ..run.checkpoints import load_checkpoint_file
from .evaluate import evaluate_scores
from .measures import ValidationMeasures
from .worker_lifetime import exit_with_parent


@dataclass
class CheckpointValidator:
    model: ModelProtocol
    split: Dataset
    settings: TrainingSettings
    pad_token_id: int
    measures: ValidationMeasures | None = None

    def evaluate_checkpoint(self, path, *, progress=None):
        if progress is not None:
            progress("Loading checkpoint weights.")
        checkpoint = load_checkpoint_file(path, mmap=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        model = self.model
        settings = self.settings
        if settings.validation_device == "npu":
            from .npu import OpenVinoNpuModel

            if getattr(model, "hierarchical", False):
                raise RuntimeError(
                    "NPU validation supports models with vocabulary logits; "
                    "use validation_device 'cpu' for hierarchical models"
                )

            model = OpenVinoNpuModel(model, settings.sequence_length, progress=progress)
            settings = replace(settings, minibatch_size=1)
        return evaluate_scores(
            model,
            self.split,
            settings,
            self.pad_token_id,
            progress=progress,
            measures=self.measures,
        )


_worker: CheckpointValidator | None = None
_events = None


def init_validation_worker(
    config,
    split,
    settings,
    pad_token_id,
    parent_pid,
    events,
    architecture=None,
    measures=None,
):
    global _worker, _events
    exit_with_parent(parent_pid)
    _events = events
    torch.set_num_threads(settings.validation_cpu_threads)
    # Spawn starts a fresh registry; carry the resolved parent hooks explicitly.
    architecture = architecture or get_architecture(config.architecture)
    _worker = CheckpointValidator(
        architecture.build(config).to("cpu").eval(),
        split,
        replace(settings, use_bf16=False),
        pad_token_id,
        measures,
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
    scores = _worker.evaluate_checkpoint(path, progress=progress)
    loss = None if scores is None else scores["loss"]
    bpb = None if scores is None else scores["bits_per_byte"]
    progress(
        f"Worker finished in {monotonic() - started:.1f}s; loss {loss}"
        + (f", {bpb:.6f} bits/byte" if bpb is not None else "")
        + "; result ready for training."
    )
    return scores

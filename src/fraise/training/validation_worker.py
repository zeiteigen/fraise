"""Initialize one CPU evaluator in a spawned validation process."""

from dataclasses import dataclass, replace

import torch
from datasets import Dataset

from fraise.config.training import TrainingSettings
from fraise.model.model import FraiseModel

from .checkpoints import load_checkpoint_file
from .validation import evaluate


@dataclass
class CpuValidator:
    model: FraiseModel
    split: Dataset
    settings: TrainingSettings
    pad_token_id: int

    def evaluate_checkpoint(self, path):
        checkpoint = load_checkpoint_file(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        return evaluate(self.model, self.split, self.settings, self.pad_token_id)


_worker: CpuValidator | None = None


def init_validation_worker(config, split, settings, pad_token_id):
    global _worker
    torch.set_num_threads(1)
    _worker = CpuValidator(
        FraiseModel(config).to("cpu").eval(),
        split,
        replace(settings, use_bf16=False),
        pad_token_id,
    )


def evaluate_checkpoint(path):
    if _worker is None:
        raise RuntimeError("validation worker was not initialized")
    return _worker.evaluate_checkpoint(path)

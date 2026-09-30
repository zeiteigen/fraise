"""Canonical run identity and separately recorded execution provenance."""

import json
import platform
from dataclasses import asdict

import torch

from fraise.config.training import TrainingSettings
from fraise.data.provenance import describe_dataset
from fraise.helpers.get_git_commit import get_git_snapshot
from fraise.tokenizer.loading import tokenizer_fingerprint


def run_manifest(data, tokenizer, model_config, settings, device) -> dict:
    identity = {
        "model_config": asdict(model_config),
        "training": asdict(settings),
        "token_budget_scope": "per_epoch",
        "dataset": {
            "source_order": list(data.train),
            "train": {
                name: describe_dataset(source) for name, source in data.train.items()
            },
            "validation": describe_dataset(data.validation)
            if data.validation is not None
            else None,
        },
        "tokenizer": {
            "fingerprint": tokenizer_fingerprint(tokenizer),
            "vocab_size": tokenizer.get_vocab_size(),
        },
    }
    return json.loads(
        json.dumps(
            {
                "manifest_version": 2,
                "identity": identity,
                "provenance": {
                    "git": get_git_snapshot(),
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "device": str(device),
                    "torch_compile": True,
                    "precision": "bf16-mixed" if settings.use_bf16 else "fp32",
                },
            }
        )
    )


def check_resume_identity(saved: dict, current: dict):
    if saved.get("manifest_version") != 2:
        raise ValueError(
            "legacy run manifest requires migration before training can resume"
        )
    before, after = saved["identity"], current["identity"]
    if (
        before.get("token_budget_scope", "total")
        != after.get("token_budget_scope", "total")
        and max(before["training"]["num_epochs"], after["training"]["num_epochs"]) > 1
    ):
        raise ValueError(
            "resume token budget scope does not match the saved run; "
            "start a fresh run to use per-epoch budgets with a new learning-rate schedule"
        )
    for name in ("model_config", "training", "dataset", "tokenizer"):
        previous = before[name]
        if name == "training":
            previous = dict(previous)
            if "shuffle" in previous:
                previous["shuffle_per_epoch"] = previous.pop("shuffle")
            previous.setdefault("shuffle_per_epoch", TrainingSettings.shuffle_per_epoch)
            previous.setdefault("shuffle_data", TrainingSettings.shuffle_data)
            previous.setdefault(
                "tied_weight_init_std", TrainingSettings.tied_weight_init_std
            )
        if previous != after[name]:
            raise ValueError(f"resume {name} does not match the saved run")

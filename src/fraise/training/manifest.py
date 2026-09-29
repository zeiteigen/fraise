"""Canonical run identity and separately recorded execution provenance."""

import json
import platform
from dataclasses import asdict

import torch

from fraise.data.provenance import describe_dataset
from fraise.helpers.get_git_commit import get_git_snapshot
from fraise.tokenizer.loading import tokenizer_fingerprint


def run_manifest(data, tokenizer, model_config, settings, device) -> dict:
    identity = {
        "model_config": asdict(model_config),
        "training": asdict(settings),
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
    for name in ("model_config", "training", "dataset", "tokenizer"):
        if before[name] != after[name]:
            raise ValueError(f"resume {name} does not match the saved run")

"""Canonical run identity and separately recorded execution provenance."""

import json
import platform
from dataclasses import asdict, fields

import torch

from fraise.architectures import model_config_from_dict
from fraise.config.training import TrainingSettings
from fraise.data.provenance import describe_dataset
from fraise.helpers.get_git_commit import get_git_snapshot
from fraise.tokenizer.loading import tokenizer_fingerprint
from fraise.training.optimizer import OPTIMIZER_IDENTITY

# Settings that change resource use, reporting, or checkpoint cadence but not the
# sequence of optimizer updates. They may change when a run resumes.
OPERATIONAL_SETTINGS = frozenset(
    {
        "checkpoint_interval_steps",
        "checkpoint_keep_last",
        "compile_threads",
        "log_interval_steps",
        "tokenization_batch_size",
        "validation_cpu_threads",
        "validation_device",
        "validation_timeout_seconds",
    }
)


def run_manifest(data, tokenizer, model_config, settings, device) -> dict:
    identity = {
        "optimizer": OPTIMIZER_IDENTITY,
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
                "manifest_version": 3,
                "identity": identity,
                "provenance": {
                    "git": get_git_snapshot(),
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "device": str(device),
                    "torch_compile": settings.use_compile,
                    "deterministic": settings.deterministic,
                    "attention_backend": "math" if settings.deterministic else "auto",
                    "operational_settings": {
                        name: getattr(settings, name)
                        for name in sorted(OPERATIONAL_SETTINGS)
                    },
                    "precision": "bf16-mixed" if settings.use_bf16 else "fp32",
                    "float32_matmul_precision": "high"
                    if device.type == "cuda" and not settings.deterministic
                    else "highest",
                    "compiled_region": "forward+loss" if settings.use_compile else None,
                },
            }
        )
    )


def check_resume_identity(saved: dict, current: dict):
    if saved.get("manifest_version") != 3:
        raise ValueError(
            "run manifest is not version 3 (single-pass training); start a fresh run"
        )
    before, after = saved["identity"], current["identity"]
    if before.get("optimizer") != after["optimizer"]:
        raise ValueError(
            "resume optimizer does not match the saved run; start a fresh run"
        )
    for name in ("model_config", "training", "dataset", "tokenizer"):
        previous = before[name]
        current = after[name]
        if name == "training":
            previous = _training_identity(previous)
            current = _training_identity(current)
        elif name == "model_config":
            previous = asdict(model_config_from_dict(previous))
            current = asdict(model_config_from_dict(current))
        if previous != current:
            raise ValueError(f"resume {name} does not match the saved run")


def _training_identity(values: dict) -> dict:
    """Normalize saved settings through today's defaults."""
    known = {field.name for field in fields(TrainingSettings)}
    settings = TrainingSettings.from_dict(
        {key: value for key, value in values.items() if key in known}
    )
    return {
        key: value
        for key, value in asdict(settings).items()
        if key not in OPERATIONAL_SETTINGS
    }

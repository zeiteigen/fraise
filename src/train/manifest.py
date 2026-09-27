"""Provenance recorded alongside each training checkpoint."""

from collections.abc import Mapping
from dataclasses import asdict
import platform

import torch
from datasets import DatasetDict
from tokenizers import Tokenizer

from components import device
from datasets_mgr import get_dataset_fingerprint, get_dataset_name, get_dataset_size_bytes
from helpers.get_git_commit import get_git_snapshot
from tokenizer import get_tokenizer_fingerprint

from .hyperparameters import TrainingSettings
from .packing import special_token_id


def _dataset_manifest(dataset: DatasetDict) -> dict[str, object]:
    splits: dict[str, object] = {}
    for split_name, split_dataset in sorted(dataset.items()):
        splits[split_name] = {
            "rows": len(split_dataset),
            "size_in_bytes": get_dataset_size_bytes(split_dataset),
            "fingerprint": get_dataset_fingerprint(split_dataset),
        }
    return {
        "name": get_dataset_name(dataset),
        "fingerprint": get_dataset_fingerprint(dataset),
        "splits": splits,
    }


def _run_manifest(
    dataset: DatasetDict,
    tokenizer: Tokenizer,
    model_config: Mapping[str, object],
    settings: TrainingSettings,
) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "git": get_git_snapshot(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "torch_compile": True,
        "precision": "bf16-mixed" if settings.use_bf16 else "fp32",
        "model_config": dict(model_config),
        "training": asdict(settings),
        "dataset": _dataset_manifest(dataset),
        "tokenizer": {
            "fingerprint": get_tokenizer_fingerprint(tokenizer),
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token_id": special_token_id(tokenizer, "<|pad|>"),
            "bos_token_id": special_token_id(tokenizer, "<|bos|>"),
            "eos_token_id": special_token_id(tokenizer, "<|eos|>"),
        },
    }

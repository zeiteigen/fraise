"""Explicitly migrate a legacy run's metadata without rewriting its weights."""

from dataclasses import asdict
from pathlib import Path

import torch

from fraise.config.model import ModelConfig
from fraise.config.training import TrainingSettings
from fraise.io import atomic_write, read_json, write_json
from fraise.tokenizer.loading import load_tokenizer, tokenizer_fingerprint

from .artifacts import META_FILE_NAME
from .manifest import run_manifest


def migrate_run(directory: Path, data):
    path = directory / META_FILE_NAME
    metadata = read_json(path)
    old = metadata["manifest"]
    if old.get("manifest_version") == 2:
        return
    config = ModelConfig(**old["model_config"])
    settings = TrainingSettings.from_dict(old["training"])
    tokenizer = load_tokenizer(directory / metadata["tokenizer_file"])
    if tokenizer_fingerprint(tokenizer) != old["tokenizer"]["fingerprint"]:
        raise ValueError("tokenizer differs from the recorded legacy run")
    if settings.dataset_token_counts.keys() - (
        data.train.keys()
        | {name for name, count in settings.dataset_token_counts.items() if count == 0}
    ):
        raise ValueError("migration data is missing allocated training sources")
    atomic_write(
        directory / "model_meta.v2.json", lambda handle: handle.write(path.read_text())
    )
    metadata["legacy_manifest"] = old
    metadata["manifest"] = run_manifest(
        data, tokenizer, config, settings, torch.device("cpu")
    )
    # Metadata migration must preserve the original training budget semantics.
    metadata["manifest"]["identity"]["token_budget_scope"] = old.get(
        "token_budget_scope", "total"
    )
    metadata["schema_version"] = 3
    messages = metadata.pop("all_messages", [])
    if messages:
        atomic_write(
            directory / "legacy_training.log",
            lambda handle: handle.write("\n".join(messages) + "\n"),
        )
    write_json(directory / "model_config.json", asdict(config))
    write_json(directory / "training_hyperparameters.json", asdict(settings))
    write_json(path, metadata)

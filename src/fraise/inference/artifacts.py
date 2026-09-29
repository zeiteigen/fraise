"""Load a run's authoritative configuration, tokenizer, and model weights."""

from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from fraise.config.model import ModelConfig
from fraise.io import read_json
from fraise.model.model import FraiseModel
from fraise.tokenizer.loading import load_tokenizer, tokenizer_fingerprint
from fraise.training.artifacts import META_FILE_NAME, resolve_checkpoint
from fraise.training.checkpoints import load_model_weights


@dataclass
class LoadedRun:
    model: FraiseModel
    tokenizer: Tokenizer
    config: ModelConfig
    path: Path


def list_model_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in Path(root).glob(f"*/{META_FILE_NAME}"))


def load_run(
    directory: str | Path, *, checkpoint: str | Path | None = None, device=None
) -> LoadedRun:
    directory = Path(directory).expanduser()
    metadata = read_json(directory / META_FILE_NAME)
    manifest = metadata["manifest"]
    identity = (
        manifest["identity"] if manifest.get("manifest_version") == 2 else manifest
    )
    config = ModelConfig(**identity["model_config"])
    tokenizer = load_tokenizer(directory / metadata["tokenizer_file"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")
    if tokenizer_fingerprint(tokenizer) != identity["tokenizer"]["fingerprint"]:
        raise ValueError("tokenizer does not match the saved run")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = directory / path
    elif metadata.get("training_finished"):
        path = directory / f"{metadata['model_name']}.pt"
    else:
        path = resolve_checkpoint(directory)
    model = FraiseModel(config)
    model.load_state_dict(load_model_weights(path))
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return LoadedRun(model, tokenizer, config, path)

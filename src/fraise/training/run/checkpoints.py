"""Serialize, restore, and retain resumable checkpoints."""

import pickle
from pathlib import Path

import numpy as np
import torch

from fraise.io import atomic_write

from .rng import capture_rng_state, restore_rng_state

SCHEMA_VERSION = 4
# Schema 2 and 3 checkpoints pickled NumPy's RNG state; allow only those types.
_LEGACY_NUMPY_GLOBALS = [
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.UInt32DType,
]


def checkpoint_step(path: Path) -> int | None:
    step = path.stem.removeprefix("checkpoint_step_")
    return int(step) if path.stem != step and step.isdecimal() else None


def step_checkpoints(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.glob("checkpoint_step_*.pt")
        if checkpoint_step(path) is not None
    ]
    return sorted(paths, key=checkpoint_step)


def save_model_state_dict(state_dict, path: str | Path) -> Path:
    return atomic_write(
        Path(path), lambda handle: torch.save(state_dict, handle), binary=True
    )


def migrate_v2_weights(weights: dict) -> dict:
    """Translate the old wrapper names at the checkpoint-format boundary."""
    return {
        name.replace("token_embedding.embedding.", "token_embedding.")
        .replace("lm_head.projection.", "lm_head.")
        .replace(".swiGLU.", ".swiglu."): value
        for name, value in weights.items()
    }


def _validate_checkpoint(checkpoint, path) -> dict:
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") not in (
        2,
        3,
        SCHEMA_VERSION,
    ):
        raise ValueError(f"unsupported training checkpoint: {path}")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "training_state",
        "rng_state",
    }
    if missing := required - checkpoint.keys():
        raise ValueError(f"checkpoint is missing {', '.join(sorted(missing))}: {path}")
    for name in required:
        if not isinstance(checkpoint[name], dict):
            raise ValueError(f"checkpoint {name} must be a dictionary: {path}")
    if checkpoint["schema_version"] == 2:
        checkpoint["training_state"].pop("epoch_tokens_seen", None)
        checkpoint["model_state_dict"] = migrate_v2_weights(
            checkpoint["model_state_dict"]
        )
    return checkpoint


def _safe_load(path: str | Path, *, mmap: bool = False):
    """Load tensors and plain containers only; never execute pickled code."""
    try:
        with torch.serialization.safe_globals(_LEGACY_NUMPY_GLOBALS):
            return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"refusing to load untrusted objects from {path}: {error}"
        ) from error


def load_checkpoint_file(path: str | Path, *, mmap: bool = False) -> dict:
    return _validate_checkpoint(_safe_load(path, mmap=mmap), path)


def load_model_weights(path: str | Path, *, migrate_legacy: bool = True) -> dict:
    weights = _safe_load(path)
    if isinstance(weights, dict) and "schema_version" in weights:
        return _validate_checkpoint(weights, path)["model_state_dict"]
    if (
        not isinstance(weights, dict)
        or not weights
        or any(not isinstance(value, torch.Tensor) for value in weights.values())
    ):
        raise ValueError(f"expected model weights: {path}")
    return migrate_v2_weights(weights) if migrate_legacy else weights


class CheckpointStore:
    def __init__(self, directory: Path, latest: Path | None = None):
        self.directory = directory
        self.latest = latest

    def save(self, model, optimizer, scheduler, state) -> Path:
        path = self.directory / f"checkpoint_step_{state.global_step:08d}.pt"
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_state": state.as_checkpoint(),
            "rng_state": capture_rng_state(),
        }
        save_model_state_dict(checkpoint, path)
        self.latest = path
        return path

    def restore(self, model, optimizer, scheduler):
        from .state import TrainingState

        if self.latest is None:
            return TrainingState()
        checkpoint = load_checkpoint_file(self.latest)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        return TrainingState.from_checkpoint(checkpoint)

    def supersede_after(self, step: int) -> tuple[Path, list[Path]] | None:
        """Move checkpoints from an abandoned continuation out of the live run.

        Resuming from an older checkpoint starts a new timeline. Later step files
        would otherwise survive retention (which keeps the highest steps) and be
        chosen by a later directory resume. They are moved, never deleted.
        """
        newer = [
            path
            for path in step_checkpoints(self.directory)
            if checkpoint_step(path) > step
        ]
        if not newer:
            return None
        base = self.directory / f"superseded-after-step-{step:08d}"
        target, suffix = base, 1
        while target.exists():
            suffix += 1
            target = base.with_name(f"{base.name}-{suffix}")
        target.mkdir()
        for path in newer:
            path.rename(target / path.name)
        return target, newer

    def prune(self, keep_last: int, protected=()):
        paths = step_checkpoints(self.directory)
        retained = set(paths[-keep_last:]) | set(protected) | {self.latest}
        for path in paths:
            if path not in retained:
                path.unlink()


def checkpoint_to_model(checkpoint_path: Path, model_path: Path) -> Path:
    """Write checkpoint weights in the final model weight format."""
    if checkpoint_path.resolve() == model_path.resolve():
        raise ValueError("model path must differ from checkpoint path")

    checkpoint = load_checkpoint_file(checkpoint_path)
    return save_model_state_dict(checkpoint["model_state_dict"], model_path)

"""Serialize, restore, and retain resumable checkpoints."""

from pathlib import Path

import torch

from fraise.io import atomic_write

from .rng import capture_rng_state, restore_rng_state

SCHEMA_VERSION = 3


def step_checkpoints(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.glob("checkpoint_step_*.pt")
        if path.stem.removeprefix("checkpoint_step_").isdecimal()
    ]
    return sorted(
        paths, key=lambda path: int(path.stem.removeprefix("checkpoint_step_"))
    )


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


def load_checkpoint_file(path: str | Path) -> dict:
    return _validate_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False), path
    )


def load_model_weights(path: str | Path) -> dict:
    weights = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(weights, dict) and "schema_version" in weights:
        return _validate_checkpoint(weights, path)["model_state_dict"]
    if (
        not isinstance(weights, dict)
        or not weights
        or any(not isinstance(value, torch.Tensor) for value in weights.values())
    ):
        raise ValueError(f"expected model weights: {path}")
    return migrate_v2_weights(weights)


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

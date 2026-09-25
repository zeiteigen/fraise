"""Crash-resilient checkpoint and run-manifest management."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import random
import time

from helpers.get_git_commit import get_git_commit

import torch

META_FILE_NAME = "model_meta.json"
CHECKPOINT_SCHEMA_VERSION = 2


def capture_rng_state() -> dict[str, object]:
    """Capture RNG state needed to continue a run reproducibly."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state["numpy"] = np.random.get_state()

    return state


def restore_rng_state(state: Mapping[str, object] | None) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)

    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, (list, tuple)):
        torch.cuda.set_rng_state_all(list(cuda_state))

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.set_state(numpy_state)


class SaveManager:
    """Persist a run manifest and resumable step checkpoints.

    A fresh run refuses to reuse a non-empty run directory. Resuming must be
    explicit so a typo or accidental relaunch cannot overwrite an experiment.
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str,
        *,
        manifest: dict[str, object] | None = None,
        resume_from: str | Path | None = None,
    ):
        self.model_name = model_name
        self.messages: list[str] = []
        self.epochs_completed = 0
        self.latest_checkpoint: Path | None = None

        if resume_from is not None:
            resume_path = self._resolve_resume_path(Path(resume_from))
            self.save_dir = resume_path.parent
            self.metadata_path = self.save_dir / META_FILE_NAME
            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    f"checkpoint metadata not found at {self.metadata_path}"
                )
            self.metadata = self._read_metadata()
            saved_manifest = self.metadata.get("manifest")
            if manifest is not None and saved_manifest != manifest:
                raise ValueError(
                    "resume manifest does not match the checkpoint manifest; "
                    "refusing to mix data, tokenizer, or training settings"
                )
            self.messages = list(self.metadata.get("all_messages", []))
            self.epochs_completed = int(self.metadata.get("epochs_completed", 0))
            self.latest_checkpoint = resume_path
            return

        self.save_dir = Path(save_dir) / model_name
        self.metadata_path = self.save_dir / META_FILE_NAME
        if self.save_dir.exists() and any(self.save_dir.iterdir()):
            raise FileExistsError(
                f"run directory is not empty: {self.save_dir}; "
                "choose a new model_name or pass --resume"
            )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_name": self.model_name,
            "start_time": time.time(),
            "git_commit": get_git_commit(),
            "manifest": manifest or {},
            "all_messages": self.messages,
        }
        self._write_metadata()

    def _read_metadata(self) -> dict[str, object]:
        with self.metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if not isinstance(metadata, dict):
            raise ValueError(f"expected an object in {self.metadata_path}")
        return metadata

    def _write_metadata(self) -> None:
        temporary_path = self.metadata_path.with_name(
            f".{self.metadata_path.name}.tmp"
        )
        with temporary_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
        os.replace(temporary_path, self.metadata_path)

    def _resolve_resume_path(self, path: Path) -> Path:
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"resume path does not exist: {path}")

        metadata_path = path / META_FILE_NAME
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            latest = metadata.get("latest_checkpoint")
            if isinstance(latest, str):
                latest_path = path / latest
                if latest_path.exists():
                    return latest_path

        checkpoints = sorted(path.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no step checkpoints found in {path}")
        return checkpoints[-1]

    def log(self, message: str, print_to_console: bool = True) -> None:
        self.messages.append(message)
        if print_to_console:
            print(message)
        self.metadata["all_messages"] = self.messages
        self.metadata["last_updated"] = time.time()
        self._write_metadata()

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        *,
        epoch: int,
        next_batch_index: int,
        global_step: int,
        tokens_seen: int,
        epoch_tokens_seen: int,
        extra_state: dict[str, object] | None = None,
    ) -> Path:
        """Atomically save a checkpoint after a completed optimizer step."""
        checkpoint_path = self.save_dir / f"checkpoint_step_{global_step:08d}.pt"
        temporary_path = checkpoint_path.with_name(
            f".{checkpoint_path.name}.tmp"
        )

        training_state = {
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "epoch_tokens_seen": epoch_tokens_seen,
        }
        if extra_state:
            training_state.update(extra_state)

        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": (
                scaler.state_dict() if scaler is not None else None
            ),
            "training_state": training_state,
            "rng_state": capture_rng_state(),
        }
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

        self.latest_checkpoint = checkpoint_path
        # ``epoch`` is the next epoch to resume. Mid-epoch checkpoints keep
        # the current zero-based epoch; epoch-end checkpoints pass the next
        # epoch after resetting the batch cursor.
        self.epochs_completed = max(self.epochs_completed, epoch)
        self.metadata.update(
            {
                "epochs_completed": self.epochs_completed,
                "latest_checkpoint": checkpoint_path.name,
                "last_updated": time.time(),
                "global_step": global_step,
                "tokens_seen": tokens_seen,
                "training_state": training_state,
            }
        )
        self._write_metadata()
        return checkpoint_path

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> dict[str, object]:
        """Load model/training/RNG state and return the saved payload."""
        if self.latest_checkpoint is None:
            raise RuntimeError("no checkpoint has been selected for resuming")

        checkpoint = torch.load(
            self.latest_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported checkpoint schema: "
                f"{checkpoint.get('schema_version')!r}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint

    def finish(
        self,
        model: torch.nn.Module,
        *,
        training_state: dict[str, object] | None = None,
    ) -> Path:
        """Atomically save the final model and mark the run complete."""
        if self.latest_checkpoint is None:
            raise RuntimeError("No checkpoints have been saved.")

        final_path = self.save_dir / f"{self.model_name}.pt"
        temporary_path = final_path.with_name(f".{final_path.name}.tmp")
        torch.save(model.state_dict(), temporary_path)
        os.replace(temporary_path, final_path)

        self.metadata.update(
            {
                "training_finished": True,
                "final_model_path": str(final_path),
                "end_time": time.time(),
            }
        )
        if training_state is not None:
            self.metadata["training_state"] = training_state
        self.log("Training finished. Metadata updated.")
        return final_path

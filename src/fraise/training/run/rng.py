"""Seed and checkpoint the random number generators used in training."""

import random
from collections.abc import Mapping

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    """Capture RNG state stored with each training checkpoint."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    # Plain tensors and numbers keep checkpoints loadable with weights_only=True.
    state["numpy"] = {
        "algorithm": algorithm,
        "keys": torch.from_numpy(keys.astype(np.int64)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore the RNG payload validated by the checkpoint reader."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, Mapping):
        numpy_state = (
            numpy_state["algorithm"],
            numpy_state["keys"].numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    np.random.set_state(numpy_state)
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

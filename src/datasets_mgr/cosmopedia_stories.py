"""Cosmopedia stories dataset source."""

from pathlib import Path
from tokenizers import Tokenizer

from .hf_dataset import HFDataset


DATA_DIR = Path("data/raw/cosmopedia-stories")
DATASET_NAME = "cosmopedia-stories"
REPO_ID = "HuggingFaceTB/cosmopedia"
CONFIG = "stories"


def use_cosmopedia_stories(
    size_gb: float | None = None,
    *,
    validation_size_gb: float = 0.0,
    data_dir: str | Path = DATA_DIR,
    force_download: bool = False,
    max_train_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
):
    """Return Cosmopedia stories, optionally limited to ``size_gb`` GB."""
    return HFDataset(
        repo_id=REPO_ID,
        config=CONFIG,
        data_dir=data_dir,
        dataset_name=DATASET_NAME,
        cache_prefix="cosmopedia_stories",
    ).load(
        size_gb=size_gb,
        validation_size_gb=validation_size_gb,
        force_download=force_download,
        max_train_tokens=max_train_tokens,
        tokenizer=tokenizer,
    )


# Compatibility with the copied prototype module's public function name.
use_fineweb_edu = use_cosmopedia_stories


__all__ = ["use_cosmopedia_stories", "use_fineweb_edu"]

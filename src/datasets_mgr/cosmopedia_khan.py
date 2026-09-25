"""Cosmopedia Khan Academy dataset source."""

from pathlib import Path

from .hf_dataset import HFDataset


DATA_DIR = Path("data/raw/cosmopedia-khan-academy")
DATASET_NAME = "cosmopedia-khanacademy"
REPO_ID = "HuggingFaceTB/cosmopedia"
CONFIG = "khanacademy"


def use_cosmopedia_khan_academy(
    size_gb: float | None = None,
    *,
    validation_size_gb: float = 0.0,
    data_dir: str | Path = DATA_DIR,
    force_download: bool = False,
):
    """Return Cosmopedia Khan Academy, optionally limited to ``size_gb`` GB."""
    return HFDataset(
        repo_id=REPO_ID,
        config=CONFIG,
        data_dir=data_dir,
        dataset_name=DATASET_NAME,
        cache_prefix="cosmopedia_khan",
    ).load(
        size_gb=size_gb,
        validation_size_gb=validation_size_gb,
        force_download=force_download,
    )


# Compatibility with the copied prototype module's public function name.
use_fineweb_edu = use_cosmopedia_khan_academy


__all__ = ["use_cosmopedia_khan_academy", "use_fineweb_edu"]

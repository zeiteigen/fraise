"""BabyLM dataset source."""

from pathlib import Path

from .hf_dataset import HFDataset


DATA_DIR = Path("data/raw/babylm-2026-100m")
DATASET_NAME = "babylm"
REPO_ID = "BabyLM-community/BabyLM-2026-Strict"
REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"


def use_babylm(
    size_gb: float | None = None,
    *,
    data_dir: str | Path = DATA_DIR,
    force_download: bool = False,
):
    """Return BabyLM, or only the requested decimal-GB source portion.

    With the default ``size_gb=None`` the repository's ``*.train.txt`` files
    are downloaded in their native format. A finite size streams rows and
    materializes only that portion locally.
    """
    return HFDataset(
        repo_id=REPO_ID,
        revision=REVISION,
        data_dir=data_dir,
        dataset_name=DATASET_NAME,
        source_kind="text_files",
        file_patterns=("*.train.txt",),
    ).load(size_gb=size_gb, force_download=force_download)


__all__ = [
    "DATASET_NAME",
    "DATA_DIR",
    "REPO_ID",
    "REVISION",
    "use_babylm",
]

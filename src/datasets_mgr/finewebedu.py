"""FineWeb-Edu dataset source."""

from pathlib import Path

from .hf_dataset import HFDataset


DATA_DIR = Path("data/raw/fineweb-edu")
DATASET_NAME = "fineweb-edu"
REPO_ID = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"

# Default to a bounded training slice and a small validation slice.
# Pass size_gb=None to consume the complete source when training is included.
TRAIN_BYTES = 10_000_000_000
VALIDATION_BYTES = 50_000_000
DEFAULT_SIZE_GB = (TRAIN_BYTES + VALIDATION_BYTES) / 1_000_000_000
DEFAULT_VALIDATION_SIZE_GB = VALIDATION_BYTES / 1_000_000_000


def use_fineweb_edu(
    size_gb: float | None = DEFAULT_SIZE_GB,
    *,
    validation_size_gb: float = DEFAULT_VALIDATION_SIZE_GB,
    data_dir: str | Path = DATA_DIR,
    force_download: bool = False,
    include_train: bool = True,
):
    """Return FineWeb-Edu, optionally bounded to ``size_gb`` decimal GB."""
    if not include_train and validation_size_gb > 0:
        size_gb = (
            validation_size_gb
            if size_gb is None
            else min(size_gb, validation_size_gb)
        )
    return HFDataset(
        repo_id=REPO_ID,
        config=CONFIG,
        data_dir=data_dir,
        dataset_name=DATASET_NAME,
        cache_prefix="fineweb_edu",
    ).load(
        size_gb=size_gb,
        validation_size_gb=validation_size_gb,
        force_download=force_download,
        include_train=include_train,
    )


__all__ = [
    "CONFIG",
    "DATASET_NAME",
    "DEFAULT_SIZE_GB",
    "DEFAULT_VALIDATION_SIZE_GB",
    "REPO_ID",
    "use_fineweb_edu",
]

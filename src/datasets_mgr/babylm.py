
from pathlib import Path

from huggingface_hub import snapshot_download

from datasets import load_dataset

DATA_DIR = Path("data/raw/babylm-2026-100m")

REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"

def use_babylm():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
    repo_id="BabyLM-community/BabyLM-2026-Strict",
    repo_type="dataset",
    revision=REVISION,
    local_dir=DATA_DIR,
    allow_patterns=["*.train.txt"],
   )
    return load_dataset(
    "text",
    data_files=str(DATA_DIR / "*.train.txt"),
    split="train",
)
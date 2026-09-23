from datasets import load_dataset
import json
from pathlib import Path


DATA_DIR = "data/raw/fineweb-edu"
DATA_FILE = "fineweb_edu_sample.jsonl"
FULL_DATA_FILE_PATH = Path(DATA_DIR) / DATA_FILE
TARGET_BYTES = 1_000_000_000  # start with ~1 GB raw text



def use_fineweb_edu():
    if not Path(DATA_DIR).exists():
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    if not (FULL_DATA_FILE_PATH).exists():
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
        )

        written = 0

        with open(FULL_DATA_FILE_PATH, "w") as f:
            for row in ds:
                text = row["text"]

                f.write(json.dumps({"text": text}) + "\n")
                written += len(text.encode("utf-8"))

                if written >= TARGET_BYTES:
                    break

        print(f"Saved {written / 1e9:.2f} GB")
    return load_dataset(
        "json",
        data_files=str(FULL_DATA_FILE_PATH),
        split="train",
    )
        

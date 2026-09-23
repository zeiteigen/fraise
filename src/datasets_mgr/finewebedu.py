from datasets import load_dataset
import json
from pathlib import Path


DATA_DIR = "data/raw/fineweb-edu"
DATASET_NAME = "fineweb-edu"
TRAIN_DATA_FILE = "fineweb_edu_train.jsonl"
VALIDATION_DATA_FILE = "fineweb_edu_validation.jsonl"
TRAIN_DATA_FILE_PATH = Path(DATA_DIR) / TRAIN_DATA_FILE
VALIDATION_DATA_FILE_PATH = Path(DATA_DIR) / VALIDATION_DATA_FILE
TRAIN_BYTES = 1_000_000_000  # start with ~1 GB raw text
VALIDATION_BYTES = 10_000_000  # start with ~10 MB raw text

def use_fineweb_edu():
    if not Path(DATA_DIR).exists():
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    if not TRAIN_DATA_FILE_PATH.exists() or not VALIDATION_DATA_FILE_PATH.exists():
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split="train",
            streaming=True,
        )

        written = 0

        with open(VALIDATION_DATA_FILE_PATH, "w") as validation_file, open(
            TRAIN_DATA_FILE_PATH, "w"
        ) as train_file:
            for row in ds:
                text = row["text"]
                record = json.dumps({"text": text}) + "\n"

                if written < VALIDATION_BYTES:
                    validation_file.write(record)
                else:
                    train_file.write(record)

                written += len(text.encode("utf-8"))

                if written >= TRAIN_BYTES + VALIDATION_BYTES:
                    break

        print(f"Saved {written / 1e9:.2f} GB")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_DATA_FILE_PATH),
            "validation": str(VALIDATION_DATA_FILE_PATH),
        },
    )
    for split_dataset in dataset.values():
        split_dataset.info.dataset_name = DATASET_NAME
    return dataset
        

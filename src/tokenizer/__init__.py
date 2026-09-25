from collections.abc import Iterator
from datetime import datetime, timezone
import json
from pathlib import Path
import re


from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


from helpers.get_git_commit import get_git_commit


from datasets_mgr import (
    get_all_dataset,
    get_dataset_name,
    get_dataset_size_bytes,
)


TOKENIZER_DIR = Path("data/tokenizers")
TOKENIZER_METADATA_FILE = TOKENIZER_DIR / "metadata.json"

BATCH_SIZE = 1_000
TEXT_COLUMN = "text"

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
]



def _iter_text_batches(
    dataset: Dataset,
    *,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
) -> Iterator[list[str]]:
    """
    Yield batches of strings without materializing the whole dataset.
    """
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Column {text_column!r} not found. "
            f"Available columns: {dataset.column_names}"
        )

    # Avoid decoding unrelated columns from the Arrow dataset.
    text_dataset = dataset.select_columns(text_column)

    for batch in text_dataset.iter(batch_size=batch_size):
        texts = [
            text
            for text in batch[text_column]
            if isinstance(text, str) and text
        ]

        if texts:
            yield texts


def _train_tokenizer(
    dataset: Dataset | DatasetDict,
    vocab_size: int,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer for ``dataset``.

    The tokenizer vocabulary size is supplied by the caller so that the
    same dataset can have multiple persisted tokenizer variants.
    """
    train_dataset = get_all_dataset(dataset)

    tokenizer = Tokenizer(BPE())

    # Conventional GPT-style byte-level pre-tokenization.
    #
    # This constrains the requested base vocabulary to normal subword-like
    # units. It does NOT constrain the later macro tokenizer, which
    # operates over the resulting token-ID sequence.
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,

        # Ensure all 256 byte values are representable, eliminating
        # the need for an <unk> token.
        initial_alphabet=ByteLevel.alphabet(),

        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _iter_text_batches(train_dataset),
        trainer=trainer,
        length=len(train_dataset),
    )

    return tokenizer




def _read_metadata() -> list[dict[str, object]]:
    """Read tokenizer metadata, accepting both list and wrapped-list formats."""
    if not TOKENIZER_METADATA_FILE.exists():
        return []

    with TOKENIZER_METADATA_FILE.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    if isinstance(metadata, dict):
        metadata = metadata.get("tokenizers")

    if not isinstance(metadata, list) or not all(
        isinstance(record, dict) for record in metadata
    ):
        raise ValueError(
            f"Expected {TOKENIZER_METADATA_FILE} to contain a list of tokenizer records."
        )

    return metadata


def _write_metadata(metadata: list[dict[str, object]]) -> None:
    TOKENIZER_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TOKENIZER_METADATA_FILE.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")


def _safe_filename_component(value: str | None) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "unknown")
    return component.strip(".-") or "unknown"


def _tokenizer_path(vocab_size: int, dataset_name: str | None) -> Path:
    dataset_component = _safe_filename_component(dataset_name)
    return TOKENIZER_DIR / f"fraise-{vocab_size}-{dataset_component}.json"


def _record_path(
    record: dict[str, object],
    vocab_size: int,
    dataset_name: str | None,
) -> Path:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        return _tokenizer_path(vocab_size, dataset_name)

    path = Path(path_value)
    return path if path.is_absolute() else TOKENIZER_DIR / path


def _find_metadata_record(
    metadata: list[dict[str, object]],
    vocab_size: int,
    dataset_name: str | None,
    dataset_size_bytes: int | None = None,
    *,
    dataset_was_supplied: bool,
) -> dict[str, object] | None:
    """Find a metadata record, rejecting ambiguous lookups without a dataset."""
    vocab_matches = [
        record
        for record in metadata
        if record.get("vocab_size") == vocab_size
    ]

    if dataset_was_supplied:
        matches = [
            record
            for record in vocab_matches
            if (
                record.get("dataset") == dataset_name
                and record.get("dataset_size_bytes") == dataset_size_bytes
            )
        ]
    else:
        matches = vocab_matches

    if len(matches) > 1:
        raise ValueError(
            "More than one tokenizer matches the requested vocabulary size. "
            "Supply the dataset used to train it."
        )

    return matches[0] if matches else None


def load_tokenizer(
    vocab_size: int,
    dataset: Dataset | DatasetDict | None = None,
) -> Tokenizer:
    """
    Load or train a tokenizer for ``vocab_size`` and ``dataset``.

    Existing tokenizers are selected using their requested vocabulary size,
    dataset name, and dataset size in bytes.  When no dataset is supplied, a
    tokenizer can still be loaded if exactly one metadata record has the
    requested vocabulary size.

    If no matching tokenizer exists, ``dataset`` is required so a new one
    can be trained.  Metadata for every trained tokenizer is kept in
    ``data/tokenizers/metadata.json``.
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")

    dataset_name = (
        get_dataset_name(dataset) if dataset is not None else None
    )
    dataset_size_bytes = (
        get_dataset_size_bytes(dataset) if dataset is not None else None
    )
    metadata = _read_metadata()
    metadata_record = _find_metadata_record(
        metadata,
        vocab_size,
        dataset_name,
        dataset_size_bytes,
        dataset_was_supplied=dataset is not None,
    )

    if metadata_record is not None:
        tokenizer_path = _record_path(
            metadata_record,
            vocab_size,
            dataset_name,
        )
        if tokenizer_path.exists():
            return Tokenizer.from_file(str(tokenizer_path))

    if dataset is None:
        raise FileNotFoundError(
            "No tokenizer found for "
            f"vocab_size={vocab_size} and dataset={dataset_name!r}; "
            "a dataset is required to train one."
        )

    tokenizer = _train_tokenizer(dataset, vocab_size)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = _tokenizer_path(vocab_size, dataset_name)
    tokenizer.save(str(tokenizer_path))

    new_record: dict[str, object] = {
        "path": str(tokenizer_path.relative_to(TOKENIZER_DIR)),
        "vocab_size": vocab_size,
        "git_commit": get_git_commit(),
        "dataset": dataset_name,
        "dataset_size_bytes": dataset_size_bytes,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    if metadata_record is not None:
        metadata[metadata.index(metadata_record)] = new_record
    else:
        metadata.append(new_record)
    _write_metadata(metadata)

    # Reload the persisted tokenizer rather than returning the in-memory
    # instance, ensuring callers always receive exactly what is on disk.
    return Tokenizer.from_file(str(tokenizer_path))

from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


from helpers.get_git_commit import get_git_commit


from datasets_mgr import (
    get_dataset_name,
    get_dataset_fingerprint,
    get_dataset_split,
    get_dataset_size_bytes,
    get_dataset_sources,
)


# Resolve the cache relative to the repository rather than the process's
# current working directory.  CLI entry points and notebooks can be launched
# from outside the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizers"
TOKENIZER_METADATA_FILE = TOKENIZER_DIR / "metadata.json"

BATCH_SIZE = 1_000
TEXT_COLUMN = "text"

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
]

TOKENIZER_SCHEMA_VERSION = 2
TOKENIZER_MIN_FREQUENCY = 2
ESTIMATED_BYTES_PER_TOKEN = 4
BYTES_PER_GB = 1_000_000_000


def get_tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    """Hash the complete serialized tokenizer, including post-processing."""
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as tokenizer_file:
        for chunk in iter(lambda: tokenizer_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Install the explicit document boundary behavior used by training."""
    missing_tokens = [
        token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None
    ]
    if missing_tokens:
        tokenizer.add_special_tokens(missing_tokens)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer must contain both BOS and EOS tokens")

    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> <|bos|> $B <|eos|>",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )
    return tokenizer



def _iter_text_batches(
    dataset: Dataset,
    *,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    max_bytes: int | None = None,
) -> Iterator[list[str]]:
    """
    Yield batches of strings without materializing the whole dataset.
    """
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Column {text_column!r} not found. "
            f"Available columns: {dataset.column_names}"
        )
    if max_bytes == 0:
        return

    # Avoid decoding unrelated columns from the Arrow dataset.
    text_dataset = dataset.select_columns(text_column)

    bytes_seen = 0
    for batch in text_dataset.iter(batch_size=batch_size):
        texts = []
        for text in batch[text_column]:
            if not isinstance(text, str) or not text:
                continue
            texts.append(text)
            if max_bytes is not None:
                bytes_seen += len(text.encode("utf-8"))
                if bytes_seen >= max_bytes:
                    break

        if texts:
            yield texts
        if max_bytes is not None and bytes_seen >= max_bytes:
            return


def _train_tokenizer(
    dataset: Dataset | DatasetDict,
    vocab_size: int,
    training_token_budgets: Mapping[str, int] | None = None,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer for ``dataset``.

    The tokenizer vocabulary size is supplied by the caller so that the
    same dataset can have multiple persisted tokenizer variants.
    """
    train_dataset = get_dataset_split(dataset, "train")

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
        min_frequency=TOKENIZER_MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,

        # Ensure all 256 byte values are representable, eliminating
        # the need for an <unk> token.
        initial_alphabet=ByteLevel.alphabet(),

        show_progress=True,
    )

    if training_token_budgets is None:
        text_batches = _iter_text_batches(train_dataset)
        length = len(train_dataset)
    else:
        sources = get_dataset_sources(dataset, "train")

        def budgeted_batches() -> Iterator[list[str]]:
            for name, source in sources.items():
                byte_limit = training_token_budgets[name] * ESTIMATED_BYTES_PER_TOKEN
                if byte_limit == 0:
                    continue
                print(
                    f"Training tokenizer on up to {byte_limit / BYTES_PER_GB:.3f} "
                    f"GB from {name}."
                )
                yield from _iter_text_batches(source, max_bytes=byte_limit)

        text_batches = budgeted_batches()
        length = None

    tokenizer.train_from_iterator(text_batches, trainer=trainer, length=length)

    return _configure_tokenizer(tokenizer)




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
    dataset_fingerprint: str | None = None,
    training_token_budgets: Mapping[str, int] | None = None,
    *,
    dataset_was_supplied: bool,
) -> dict[str, object] | None:
    """Find a metadata record, rejecting ambiguous lookups without a dataset."""
    vocab_matches = [
        record
        for record in metadata
        if (
            record.get("vocab_size") == vocab_size
            and record.get("schema_version") == TOKENIZER_SCHEMA_VERSION
        )
    ]

    if dataset_was_supplied:
        matches = [
            record
            for record in vocab_matches
            if (
                record.get("dataset") == dataset_name
                and record.get("dataset_size_bytes") == dataset_size_bytes
                and record.get("dataset_fingerprint") == dataset_fingerprint
                and record.get("training_split") == "train"
                and record.get("training_token_budgets")
                == (
                    dict(training_token_budgets)
                    if training_token_budgets is not None
                    else None
                )
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


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    """Load and configure the tokenizer at the explicitly supplied path."""
    path = Path(tokenizer_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer file does not exist: {path}")
    return _configure_tokenizer(Tokenizer.from_file(str(path)))


def get_or_train_tokenizer(
    vocab_size: int,
    dataset: Dataset | DatasetDict | None = None,
    *,
    training_token_budgets: Mapping[str, int] | None = None,
) -> Tokenizer:
    """
    Reuse or train a tokenizer for ``vocab_size`` and ``dataset``.

    A cached tokenizer is reusable only when metadata matches the supplied
    training dataset.  Metadata also supports loading older/custom paths.
    When no dataset is supplied, a tokenizer can still be loaded if exactly
    one metadata record has the requested vocabulary size.

    If no matching tokenizer exists, ``dataset`` is required so a new one
    can be trained.  Metadata for every trained tokenizer is kept in
    ``data/tokenizers/metadata.json``.
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    if training_token_budgets is not None:
        if dataset is None:
            raise ValueError("a dataset is required with training_token_budgets")
        source_names = set(get_dataset_sources(dataset, "train"))
        if set(training_token_budgets) != source_names or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in training_token_budgets.values()
        ):
            raise ValueError(
                "training_token_budgets must give a non-negative integer "
                "for every training dataset"
            )

    dataset_name = (
        get_dataset_name(dataset) if dataset is not None else None
    )

    training_dataset = (
        get_dataset_split(dataset, "train") if dataset is not None else None
    )
    dataset_size_bytes = (
        get_dataset_size_bytes(training_dataset)
        if training_dataset is not None
        else None
    )
    dataset_fingerprint = (
        get_dataset_fingerprint(training_dataset)
        if training_dataset is not None
        else None
    )
    metadata = _read_metadata()
    metadata_record = _find_metadata_record(
        metadata,
        vocab_size,
        dataset_name,
        dataset_size_bytes,
        dataset_fingerprint,
        training_token_budgets,
        dataset_was_supplied=dataset is not None,
    )

    tokenizer_path = _tokenizer_path(vocab_size, dataset_name)
    if metadata_record is not None:
        recorded_path = _record_path(
            metadata_record,
            vocab_size,
            dataset_name,
        )
        if recorded_path.exists():
            expected_sha256 = metadata_record.get("tokenizer_sha256")
            checksum_matches = (
                not isinstance(expected_sha256, str)
                or _sha256_file(recorded_path) == expected_sha256
            )
            if checksum_matches:
                return load_tokenizer(recorded_path)

    if dataset is None:
        raise FileNotFoundError(
            "No tokenizer found for "
            f"vocab_size={vocab_size} and dataset={dataset_name!r}; "
            "a dataset is required to train one."
        )

    if tokenizer_path.exists():
        print(
            f"No matching tokenizer metadata found for {tokenizer_path}. "
            "A new tokenizer will be trained and persisted."
        )
    else:
        print(
            f"No tokenizer found at {tokenizer_path}. "
            "A new one will be trained and persisted."
        )

    tokenizer = _train_tokenizer(dataset, vocab_size, training_token_budgets)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = _tokenizer_path(vocab_size, dataset_name)
    tokenizer.save(str(tokenizer_path))

    new_record: dict[str, object] = {
        "schema_version": TOKENIZER_SCHEMA_VERSION,
        "path": str(tokenizer_path.relative_to(TOKENIZER_DIR)),
        "vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "git_commit": get_git_commit(),
        "dataset": dataset_name,
        "dataset_size_bytes": dataset_size_bytes,
        "dataset_fingerprint": dataset_fingerprint,
        "training_split": "train",
        "training_token_budgets": (
            dict(training_token_budgets) if training_token_budgets is not None else None
        ),
        "min_frequency": TOKENIZER_MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
        "post_processor": "bos_eos",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    new_record["tokenizer_sha256"] = _sha256_file(tokenizer_path)

    # The canonical filename is stable for (vocabulary size, dataset name).
    # If that file is being replaced after a dataset change, discard stale
    # records that point to it so they cannot claim the new bytes belong to
    # the old dataset fingerprint.  Also replace an exact record whose file
    # disappeared or failed its checksum.
    replace_indices = set()
    if metadata_record is not None:
        replace_indices.add(metadata.index(metadata_record))

    canonical_path = tokenizer_path.resolve()
    for index, record in enumerate(metadata):
        if (
            record.get("vocab_size") == vocab_size
            and record.get("dataset") == dataset_name
            and record.get("training_split") == "train"
            and _record_path(record, vocab_size, dataset_name).resolve()
            == canonical_path
        ):
            replace_indices.add(index)

    if replace_indices:
        insert_at = min(replace_indices)
        metadata = [
            record
            for index, record in enumerate(metadata)
            if index not in replace_indices
        ]
        metadata.insert(insert_at, new_record)
    else:
        metadata.append(new_record)
    _write_metadata(metadata)

    # Reload the persisted tokenizer rather than returning the in-memory
    # instance, ensuring callers always receive exactly what is on disk.
    return load_tokenizer(tokenizer_path)

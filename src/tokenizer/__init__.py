from collections.abc import Iterator
from pathlib import Path
import warnings

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


TOKENIZER_PATH = Path("data/tokenizers/fraise-8k.json")

VOCAB_SIZE = 8_192
BATCH_SIZE = 1_000
TEXT_COLUMN = "text"

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
]


def _resolve_dataset(
    dataset: Dataset | DatasetDict,
) -> Dataset:
    """
    Resolve Dataset | DatasetDict into the Dataset used for training.

    Dataset:
        Used directly.

    DatasetDict:
        Uses the "train" split.
    """
    if isinstance(dataset, Dataset):
        return dataset

    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(
                "DatasetDict has no 'train' split. "
                f"Available splits: {list(dataset.keys())}"
            )

        return dataset["train"]

    raise TypeError(
        "Expected datasets.Dataset or datasets.DatasetDict, "
        f"got {type(dataset).__name__}"
    )


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
) -> Tokenizer:
    """
    Train Fraise's conventional 8k base tokenizer.

    The later 64k macro tokenizer should operate over sequences of
    token IDs produced by this tokenizer.
    """
    train_dataset = _resolve_dataset(dataset)

    tokenizer = Tokenizer(BPE())

    # Conventional GPT-style byte-level pre-tokenization.
    #
    # This constrains the 8k base vocabulary to normal subword-like
    # units. It does NOT constrain the later macro tokenizer, which
    # operates over the resulting token-ID sequence.
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
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


def load_tokenizer(
    dataset: Dataset | DatasetDict | None = None,
) -> Tokenizer:
    """
    Load the saved tokenizer if it exists.

    If a saved tokenizer exists:
        - dataset is ignored
        - a warning is emitted if dataset was supplied

    Otherwise:
        - dataset is required
        - tokenizer is trained
        - tokenizer is saved
        - tokenizer is loaded back from disk and returned
    """
    if TOKENIZER_PATH.exists():
        if dataset is not None:
            warnings.warn(
                f"Tokenizer already exists at {TOKENIZER_PATH}. "
                "The supplied dataset will not be used.",
                UserWarning,
                stacklevel=2,
            )

        return Tokenizer.from_file(str(TOKENIZER_PATH))

    if dataset is None:
        raise FileNotFoundError(
            f"No tokenizer found at {TOKENIZER_PATH} and no "
            "dataset was supplied for training."
        )

    tokenizer = _train_tokenizer(dataset)

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    # Reload the persisted tokenizer rather than returning the in-memory
    # instance, ensuring callers always receive exactly what is on disk.
    return Tokenizer.from_file(str(TOKENIZER_PATH))
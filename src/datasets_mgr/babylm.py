"""BabyLM dataset source."""

from collections.abc import Iterable, Iterator
import hashlib
from pathlib import Path
import re

from datasets import Dataset, Features, Value

from .hf_dataset import HFDataset


DATA_DIR = Path("data/raw/babylm-2026-100m")
DATASET_NAME = "babylm"
REPO_ID = "BabyLM-community/BabyLM-2026-Strict"
REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"
WORDS_PER_CHUNK = 256
_WORD = re.compile(r"\S+")


def _chunk_text_rows(
    rows: Iterable[str], words_per_chunk: int
) -> Iterator[dict[str, str]]:
    """Group consecutive lines by word count while preserving their text."""
    parts: list[str] = []
    word_count = 0
    first_row = True
    completed: str | None = None

    for row in rows:
        if not first_row:
            parts.append("\n")
        first_row = False

        start = 0
        for word in _WORD.finditer(row):
            word_count += 1
            if word_count == words_per_chunk:
                parts.append(row[start : word.end()])
                if completed is not None:
                    yield {"text": completed}
                completed = "".join(parts)
                parts = []
                start = word.end()
                word_count = 0
        parts.append(row[start:])

    tail = "".join(parts)
    if word_count:
        if completed is not None:
            yield {"text": completed}
        yield {"text": tail}
    elif completed is not None:
        yield {"text": completed + tail}
    elif tail:
        yield {"text": tail}


def _chunk_dataset(source: Dataset, words_per_chunk: int) -> Dataset:
    """Cache the regrouped text as a map-style dataset for training."""
    fingerprint = hashlib.sha256(
        repr(("babylm-word-chunks-v1", source._fingerprint, words_per_chunk)).encode(
            "utf-8"
        )
    ).hexdigest()

    def rows() -> Iterator[dict[str, str]]:
        source_rows = (
            text
            for batch in source.select_columns("text").iter(batch_size=1_000)
            for text in batch["text"]
        )
        yield from _chunk_text_rows(source_rows, words_per_chunk)

    chunked = Dataset.from_generator(
        rows,
        features=Features({"text": Value("string")}),
        fingerprint=fingerprint,
        keep_in_memory=False,
    )
    chunked.info.dataset_name = source.info.dataset_name
    chunked.info.download_size = source.info.download_size
    chunked.info.dataset_size = int(chunked.data.nbytes)
    if chunked.info.download_size is not None:
        chunked.info.size_in_bytes = (
            chunked.info.download_size + chunked.info.dataset_size
        )
    return chunked


def use_babylm(
    size_gb: float | None = None,
    *,
    words_per_chunk: int = WORDS_PER_CHUNK,
    data_dir: str | Path = DATA_DIR,
    force_download: bool = False,
):
    """Return BabyLM in consecutive word chunks, or a bounded source portion.

    With the default ``size_gb=None`` the repository's ``*.train.txt`` files
    are downloaded in their native format. A finite size streams rows and
    materializes only that portion locally. Text rows are then regrouped into
    approximately ``words_per_chunk`` whitespace-delimited words while their
    original newlines are preserved.
    """
    if (
        isinstance(words_per_chunk, bool)
        or not isinstance(words_per_chunk, int)
        or words_per_chunk < 1
    ):
        raise ValueError("words_per_chunk must be a positive integer")

    source = HFDataset(
        repo_id=REPO_ID,
        revision=REVISION,
        data_dir=data_dir,
        dataset_name=DATASET_NAME,
        source_kind="text_files",
        file_patterns=("*.train.txt",),
    ).load(size_gb=size_gb, force_download=force_download)
    return _chunk_dataset(source, words_per_chunk)


__all__ = [
    "DATASET_NAME",
    "DATA_DIR",
    "REPO_ID",
    "REVISION",
    "WORDS_PER_CHUNK",
    "use_babylm",
]

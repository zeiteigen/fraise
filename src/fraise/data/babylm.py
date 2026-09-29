"""BabyLM dataset source."""

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import Dataset, Features, Value
from huggingface_hub import snapshot_download

from .arrow_cache import ArrowCache

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


def load_babylm(
    data_dir: Path, *, words_per_chunk: int = WORDS_PER_CHUNK, force_download=False
) -> Dataset:
    """Load the pinned native BabyLM snapshot and preserve passage boundaries."""
    from datasets import load_dataset

    from fraise.config.values import integer

    integer("words_per_chunk", words_per_chunk)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.glob("*.train.txt"))
    if force_download or not files:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=data_dir,
            allow_patterns=["*.train.txt"],
            force_download=force_download,
        )
        files = sorted(data_dir.glob("*.train.txt"))
    if not files:
        raise FileNotFoundError(f"no BabyLM training files in {data_dir}")
    cache = ArrowCache(
        data_dir / ".arrow-cache", {"repo_id": REPO_ID, "revision": REVISION}
    )
    source = None if force_download else cache.load("train", files)
    if source is None:
        source = load_dataset(
            "text", data_files=[str(path) for path in files], split="train"
        )
        source.info.dataset_name = DATASET_NAME
        source.info.download_size = sum(path.stat().st_size for path in files)
        source.info.dataset_size = int(source.data.nbytes)
        source.info.size_in_bytes = source.info.download_size + source.info.dataset_size
        source = cache.store("train", files, source, replace=force_download)
    return _chunk_dataset(source, words_per_chunk)

"""Download bounded Hub text prefixes and reopen their Arrow caches."""

import json
import os
import tempfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset
from filelock import FileLock
from tokenizers import Tokenizer

from fraise.config.values import integer, number
from fraise.tokenizer.loading import tokenizer_fingerprint

from .arrow_cache import ArrowCache
from .raw_cache import RawCache
from .sources import Source
from .types import TrainingData

BYTES_PER_GB = 1_000_000_000


def stream_source(source: Source):
    return load_dataset(
        source.repo_id,
        name=source.config,
        revision=source.revision,
        split="train",
        streaming=True,
    )


def download_prefixes(
    source: Source,
    caches: dict[str, RawCache],
    *,
    validation_bytes: int,
    total_bytes: int | None,
    max_train_tokens: int | None,
    tokenizer: Tokenizer | None,
):
    """Keep source rows intact; held-out rows always precede training rows."""
    temporary = {}
    try:
        with ExitStack() as stack:
            handles = {}
            for split, cache in caches.items():
                cache.directory.mkdir(parents=True, exist_ok=True)
                descriptor, name = tempfile.mkstemp(
                    prefix=".text-", dir=cache.directory
                )
                temporary[split] = Path(name)
                handles[split] = stack.enter_context(
                    os.fdopen(descriptor, "w", encoding="utf-8")
                )
            written_bytes = train_tokens = 0
            for row in stream_source(source):
                if total_bytes is not None and written_bytes >= total_bytes:
                    break
                text = row["text"]
                if not isinstance(text, str):
                    raise TypeError(f"{source.name} text must be a string")
                split = "validation" if written_bytes < validation_bytes else "train"
                if split == "train" and "train" not in handles:
                    break
                if split in handles:
                    handles[split].write(
                        json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    )
                written_bytes += len(text.encode("utf-8"))
                if split == "train" and max_train_tokens is not None:
                    train_tokens += len(tokenizer.encode(text).ids) if text else 0
                    if train_tokens >= max_train_tokens + 1:
                        break
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        for split, path in temporary.items():
            os.replace(path, caches[split].path)
            caches[split].complete()
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def load_stream(
    source: Source,
    data_dir: Path,
    *,
    size_gb: float | None = None,
    validation_size_gb: float = 0.0,
    include_train=True,
    include_validation=True,
    max_train_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
    force_download=False,
) -> TrainingData:
    if size_gb is not None:
        number("size_gb", size_gb)
    number("validation_size_gb", validation_size_gb, inclusive=True)
    if size_gb is not None and validation_size_gb > size_gb:
        raise ValueError("validation_size_gb cannot exceed size_gb")
    if max_train_tokens is not None:
        integer("max_train_tokens", max_train_tokens)
        if tokenizer is None:
            raise ValueError("tokenizer is required with max_train_tokens")
    if not include_train and (validation_size_gb == 0 or not include_validation):
        raise ValueError("validation-only loading requires a validation split")
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    validation_bytes = int(validation_size_gb * BYTES_PER_GB)
    base = {"version": 1, **asdict(source)}
    caches = {}
    if validation_bytes and include_validation:
        caches["validation"] = RawCache(
            root / "raw", {**base, "split": "validation", "bytes": validation_bytes}
        )
    if include_train:
        caches["train"] = RawCache(
            root / "raw",
            {
                **base,
                "split": "train",
                "size_gb": size_gb,
                "validation_bytes": validation_bytes,
                "max_train_tokens": max_train_tokens,
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer)
                if max_train_tokens is not None
                else None,
            },
        )
    with FileLock(str(root / ".download.lock")):
        missing = {
            split: cache
            for split, cache in caches.items()
            if force_download or not cache.ready()
        }
        if missing:
            download_prefixes(
                source,
                missing,
                validation_bytes=validation_bytes,
                total_bytes=int(size_gb * BYTES_PER_GB)
                if include_train and size_gb is not None
                else None,
                max_train_tokens=max_train_tokens,
                tokenizer=tokenizer,
            )
    result = TrainingData()
    for split, raw in caches.items():
        if raw.path.stat().st_size == 0:
            if split == "train":
                raise ValueError(
                    f"no training rows were downloaded from {source.repo_id}"
                )
            raise ValueError(
                f"no validation rows were downloaded from {source.repo_id}"
            )
        arrow = ArrowCache(root / ".arrow-cache", raw.request)
        dataset = None if force_download else arrow.load(split, [raw.path])
        if dataset is None:
            dataset = load_dataset("json", data_files=str(raw.path), split="train")
            dataset.info.dataset_name = source.name
            dataset.info.download_size = raw.path.stat().st_size
            dataset.info.dataset_size = int(dataset.data.nbytes)
            dataset.info.size_in_bytes = (
                dataset.info.download_size + dataset.info.dataset_size
            )
            dataset = arrow.store(split, [raw.path], dataset, replace=force_download)
        if split == "train":
            result.train[source.name] = dataset
        else:
            result.validation = dataset
    return result

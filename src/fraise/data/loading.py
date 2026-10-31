"""Load explicit training sources and optional FineWeb held-out data."""

from pathlib import Path

from tokenizers import Tokenizer

from fraise.paths import data_root
from fraise.tokenizer.macro import counting_tokenizer

from .sources import FINEWEB, SOURCE_NAMES, STREAM_SOURCES, VALIDATION_SIZE_GB
from .stream import load_stream
from .types import TrainingData


def load_training_data(
    counts: dict[str, int],
    tokenizer: Tokenizer,
    *,
    root: Path | None = None,
    validation=True,
    sources: set[str] | None = None,
) -> TrainingData:
    unknown = counts.keys() - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"unknown dataset name(s): {', '.join(sorted(unknown))}")
    raw_root = data_root(root) / "raw"
    result = TrainingData()
    for name in SOURCE_NAMES:
        selected = counts.get(name, 0) > 0 and (sources is None or name in sources)
        if selected or (name == FINEWEB.name and validation):
            source = STREAM_SOURCES[name]
            loaded = load_stream(
                source,
                raw_root / name,
                size_gb=source.size_gb,
                validation_size_gb=VALIDATION_SIZE_GB if name == FINEWEB.name else 0,
                include_train=selected,
                include_validation=validation,
                max_train_tokens=counts[name] if selected else None,
                # Budgets count shared base tokens, so every tokenizer built on the
                # same base selects (and caches) exactly the same source text.
                tokenizer=counting_tokenizer(tokenizer) if selected else None,
            )
            result.train.update(loaded.train)
            if validation and loaded.validation is not None:
                result.validation = loaded.validation
    return result


def load_tokenizer_sources(
    names: list[str], *, root: Path | None = None, size_gb=0.1
) -> dict:
    unknown = set(names) - set(SOURCE_NAMES)
    if unknown or not names:
        raise ValueError(f"select sources from {', '.join(SOURCE_NAMES)}")
    raw_root = data_root(root) / "raw"
    sources = {}
    for name in SOURCE_NAMES:
        if name not in names:
            continue
        # The FineWeb held-out prefix is excluded so tokenizer merges are not
        # fit to the documents that later measure validation loss.
        held_out = VALIDATION_SIZE_GB if name == FINEWEB.name else 0
        sources.update(
            load_stream(
                STREAM_SOURCES[name],
                raw_root / name,
                size_gb=size_gb + held_out,
                validation_size_gb=held_out,
                include_validation=False,
            ).train
        )
    return sources

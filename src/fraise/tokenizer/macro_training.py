"""Learn macro merges over base-token sequences with the Rust BPE trainer."""

import json
import tempfile
from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer

from fraise.config.values import integer

from .macro import (
    MAX_BASE_VOCAB_SIZE,
    MacroTokenizer,
    _segmenting_tokenizer,
    base_ids_to_text,
    boundary_base_ids,
    special_base_ids,
    text_to_base_ids,
)
from .training import MIN_FREQUENCY, iter_text_batches

DEFAULT_MAX_MACRO_LENGTH = 8


def train_macro_tokenizer(
    base: Tokenizer,
    sources: dict[str, Dataset],
    vocab_size: int,
    *,
    max_macro_length: int = DEFAULT_MAX_MACRO_LENGTH,
    max_bytes_per_source: int | None = None,
    min_frequency: int = MIN_FREQUENCY,
) -> MacroTokenizer:
    """Merge frequent adjacent base tokens until ``vocab_size`` macros exist.

    Documents are base-encoded with their BOS/EOS markers; special tokens never
    merge and line-break tokens end a macro. The trainer may stop early when the
    corpus has no further pair reaching ``min_frequency``.
    """
    base_vocab = base.get_vocab_size()
    if base_vocab > MAX_BASE_VOCAB_SIZE:
        raise ValueError(f"base vocabulary cannot exceed {MAX_BASE_VOCAB_SIZE}")
    integer("vocab_size", vocab_size, base_vocab + 1)
    integer("max_macro_length", max_macro_length, 2)
    integer("min_frequency", min_frequency)
    if max_bytes_per_source is not None:
        integer("max_bytes_per_source", max_bytes_per_source)
    if not sources or not any(len(source) for source in sources.values()):
        raise ValueError("macro tokenizer training requires non-empty text sources")

    specials = special_base_ids(base)
    boundaries = boundary_base_ids(base)
    tokenizer = _segmenting_tokenizer(base, boundaries)
    special_set = set(specials)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[base_ids_to_text([token_id]) for token_id in specials],
        initial_alphabet=[
            base_ids_to_text([token_id])
            for token_id in range(base_vocab)
            if token_id not in special_set
        ],
        limit_alphabet=base_vocab,
        # The Rust trainer only keeps merges shorter than this limit.
        max_token_length=max_macro_length + 1,
        show_progress=False,
    )

    # Base-encode before training: encoding inside the trainer's iterator runs
    # Rayon work from its reader thread, which can deadlock. The trainer then
    # reads plain strings back. Private-use characters contain no line breaks,
    # and terminators are stripped so they never become symbols.
    with tempfile.TemporaryDirectory(prefix="fraise-macro-") as directory:
        corpus = Path(directory) / "base-ids.txt"
        with corpus.open("w", encoding="utf-8") as handle:
            for source in sources.values():
                for texts in iter_text_batches(source, max_bytes=max_bytes_per_source):
                    for encoding in base.encode_batch(texts, add_special_tokens=True):
                        handle.write(base_ids_to_text(encoding.ids))
                        handle.write("\n")

        def documents():
            with corpus.open(encoding="utf-8") as lines:
                for line in lines:
                    yield line.rstrip("\n")

        tokenizer.train_from_iterator(documents(), trainer=trainer)
    model = json.loads(tokenizer.to_str())["model"]
    ids = {base_ids_to_text([token_id]): token_id for token_id in range(base_vocab)}
    merges = []
    for merge in model["merges"]:
        left, right = merge.split(" ") if isinstance(merge, str) else merge
        merges.append((ids[left], ids[right]))
        ids[left + right] = base_vocab + len(merges) - 1
    if len(ids) != base_vocab + len(merges):
        raise RuntimeError("macro trainer produced duplicate tokens")
    for token, macro_id in ids.items():
        if len(text_to_base_ids(token)) > max_macro_length:
            raise RuntimeError(f"macro {macro_id} exceeds max_macro_length")
    return MacroTokenizer(
        base, merges, max_macro_length=max_macro_length, boundary_ids=boundaries
    )

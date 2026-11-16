"""Learn macro vocabularies over base-token sequences.

``bpe`` merges frequent adjacent base tokens with the Rust BPE trainer, across
words or, for the first ``word_bounded_vocab_size`` entries, only inside words
(a SuperBPE-style curriculum when both stages run). ``greedy`` grows a
prefix-closed vocabulary by appending one base token at a time, scoring each
candidate by how often the current longest-match segmentation stops right
before that token.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from fraise.config.values import integer

from .macro import (
    MAX_BASE_VOCAB_SIZE,
    SEGMENTATIONS,
    MacroTokenizer,
    _configure_macro_stage,
    _segmenting_tokenizer,
    _word_separated_text,
    base_ids_to_text,
    boundary_base_ids,
    greedy_parse,
    special_base_ids,
    text_to_base_ids,
    word_starts_from_word_ids,
)
from .training import MIN_FREQUENCY, iter_text_batches

DEFAULT_MAX_MACRO_LENGTH = 8
DEFAULT_MERGES_PER_ROUND = 1024
# Extra stage-two merges trained so that enough survive the length filter.
_STAGE_TWO_SLACK = 4096


def train_macro_tokenizer(
    base: Tokenizer,
    sources: dict[str, Dataset],
    vocab_size: int,
    *,
    max_macro_length: int = DEFAULT_MAX_MACRO_LENGTH,
    max_bytes_per_source: int | None = None,
    min_frequency: int = MIN_FREQUENCY,
    segmentation: str = "bpe",
    word_bounded_vocab_size: int | None = None,
    word_bounded: bool = False,
    merges_per_round: int = DEFAULT_MERGES_PER_ROUND,
) -> MacroTokenizer:
    """Learn macros over ``base`` until ``vocab_size`` entries exist.

    Documents are base-encoded with their BOS/EOS markers; special tokens never
    merge and line-break tokens end a macro. Training may stop early when the
    corpus has no further candidate reaching ``min_frequency``.
    """
    base_vocab = base.get_vocab_size()
    if base_vocab > MAX_BASE_VOCAB_SIZE:
        raise ValueError(f"base vocabulary cannot exceed {MAX_BASE_VOCAB_SIZE}")
    integer("vocab_size", vocab_size, base_vocab + 1)
    integer("max_macro_length", max_macro_length, 2)
    integer("min_frequency", min_frequency)
    if max_bytes_per_source is not None:
        integer("max_bytes_per_source", max_bytes_per_source)
    if segmentation not in SEGMENTATIONS:
        raise ValueError(f"segmentation must be one of {SEGMENTATIONS}")
    if word_bounded_vocab_size is not None:
        if segmentation != "bpe":
            raise ValueError("word_bounded_vocab_size applies to bpe segmentation")
        integer("word_bounded_vocab_size", word_bounded_vocab_size, base_vocab + 1)
        word_bounded_vocab_size = min(word_bounded_vocab_size, vocab_size)
    if word_bounded and segmentation != "greedy":
        raise ValueError("word_bounded applies to greedy segmentation")
    integer("merges_per_round", merges_per_round, 1)
    if not sources or not any(len(source) for source in sources.values()):
        raise ValueError("macro tokenizer training requires non-empty text sources")

    if segmentation == "greedy":
        return _train_greedy(
            base,
            sources,
            vocab_size,
            max_macro_length=max_macro_length,
            max_bytes_per_source=max_bytes_per_source,
            min_frequency=min_frequency,
            word_bounded=word_bounded,
            merges_per_round=merges_per_round,
        )
    if word_bounded_vocab_size is None:
        return _train_cross_word_bpe(
            base, sources, vocab_size, max_macro_length, max_bytes_per_source, min_frequency
        )
    return _train_word_bounded_bpe(
        base,
        sources,
        vocab_size,
        word_bounded_vocab_size,
        max_macro_length,
        max_bytes_per_source,
        min_frequency,
    )


def _iter_base_encodings(base, sources, max_bytes_per_source):
    for source in sources.values():
        for texts in iter_text_batches(source, max_bytes=max_bytes_per_source):
            yield from base.encode_batch(texts, add_special_tokens=True)


def _bpe_trainer(vocab_size, min_frequency, specials, alphabet, max_macro_length):
    return BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[base_ids_to_text([token_id]) for token_id in specials],
        initial_alphabet=[base_ids_to_text([token_id]) for token_id in alphabet],
        limit_alphabet=len(alphabet),
        # The Rust trainer only keeps merges shorter than this limit.
        max_token_length=max_macro_length + 1,
        show_progress=False,
    )


def _train_from_lines(tokenizer, trainer, write_lines):
    # Encode before training: encoding inside the trainer's iterator runs Rayon
    # work from its reader thread, which can deadlock. The trainer then reads
    # plain strings back. Private-use characters contain no line breaks.
    with tempfile.TemporaryDirectory(prefix="fraise-macro-") as directory:
        corpus = Path(directory) / "base-ids.txt"
        with corpus.open("w", encoding="utf-8") as handle:
            write_lines(handle)

        def documents():
            with corpus.open(encoding="utf-8") as lines:
                for line in lines:
                    yield line.rstrip("\n")

        tokenizer.train_from_iterator(documents(), trainer=trainer)
    model = json.loads(tokenizer.to_str())["model"]
    return [
        tuple(merge.split(" ")) if isinstance(merge, str) else tuple(merge)
        for merge in model["merges"]
    ]


def _train_cross_word_bpe(
    base, sources, vocab_size, max_macro_length, max_bytes_per_source, min_frequency
):
    base_vocab = base.get_vocab_size()
    specials = special_base_ids(base)
    boundaries = boundary_base_ids(base)
    tokenizer = _segmenting_tokenizer(base, boundaries)
    special_set = set(specials)
    trainer = _bpe_trainer(
        vocab_size,
        min_frequency,
        specials,
        [i for i in range(base_vocab) if i not in special_set],
        max_macro_length,
    )

    def write(handle):
        for encoding in _iter_base_encodings(base, sources, max_bytes_per_source):
            handle.write(base_ids_to_text(encoding.ids))
            handle.write("\n")

    text_merges = _train_from_lines(tokenizer, trainer, write)
    ids = {base_ids_to_text([token_id]): token_id for token_id in range(base_vocab)}
    merges = []
    for left, right in text_merges:
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


def _train_word_bounded_bpe(
    base,
    sources,
    vocab_size,
    word_bounded_vocab_size,
    max_macro_length,
    max_bytes_per_source,
    min_frequency,
):
    """Word-bounded merges first; then, if room is left, merges across words."""
    base_vocab = base.get_vocab_size()
    specials = special_base_ids(base)
    special_set = set(specials)
    boundaries = boundary_base_ids(base)
    stage1 = _segmenting_tokenizer(base, boundaries, word_bounded=True)
    trainer = _bpe_trainer(
        word_bounded_vocab_size,
        min_frequency,
        specials,
        [i for i in range(base_vocab) if i not in special_set],
        max_macro_length,
    )
    sequences = []

    def write_words(handle):
        for encoding in _iter_base_encodings(base, sources, max_bytes_per_source):
            starts = word_starts_from_word_ids(encoding.word_ids)
            sequences.append((encoding.ids, starts))
            handle.write(_word_separated_text(encoding.ids, starts))
            handle.write("\n")

    ids = {base_ids_to_text([token_id]): token_id for token_id in range(base_vocab)}
    merges = []
    for left, right in _train_from_lines(stage1, trainer, write_words):
        merges.append((ids[left], ids[right]))
        ids[left + right] = base_vocab + len(merges) - 1
    split = len(merges)
    word_bounded = MacroTokenizer(
        base,
        merges,
        max_macro_length=max_macro_length,
        boundary_ids=boundaries,
        word_bounded_merges=split,
    )
    if base_vocab + split >= vocab_size:
        return word_bounded

    # Stage two: plain BPE over the stage-one IDs, each read as one character.
    alphabet = base_vocab + split
    lengths = list(word_bounded.unit_lengths)
    line_breaks = set(boundaries)
    ends_line = [
        i for i in range(alphabet) if word_bounded.decomposition(i)[-1] in line_breaks
    ]
    stage2 = Tokenizer(BPE())
    _configure_macro_stage(
        stage2, specials, base_ids_to_text(ends_line), word_bounded=False
    )
    trainer = _bpe_trainer(
        vocab_size + _STAGE_TWO_SLACK,
        min_frequency,
        specials,
        [i for i in range(alphabet) if i not in special_set],
        max_macro_length,
    )

    def write_stage_one(handle):
        batch = 4096
        for start in range(0, len(sequences), batch):
            chunk = sequences[start : start + batch]
            encoded = word_bounded._macro_ids(
                [ids_ for ids_, _ in chunk], [starts for _, starts in chunk]
            )
            for macro_ids in encoded:
                handle.write(base_ids_to_text(macro_ids))
                handle.write("\n")

    symbols = {base_ids_to_text([i]): i for i in range(alphabet)}
    dropped = set()
    for left, right in _train_from_lines(stage2, trainer, write_stage_one):
        if base_vocab + len(merges) >= vocab_size:
            break
        joined = left + right
        if left in dropped or right in dropped:
            dropped.add(joined)
            continue
        left_id, right_id = symbols[left], symbols[right]
        if lengths[left_id] + lengths[right_id] > max_macro_length:
            dropped.add(joined)
            continue
        merges.append((left_id, right_id))
        symbols[joined] = base_vocab + len(merges) - 1
        lengths.append(lengths[left_id] + lengths[right_id])
    return MacroTokenizer(
        base,
        merges,
        max_macro_length=max_macro_length,
        boundary_ids=boundaries,
        word_bounded_merges=split,
    )


def _train_greedy(
    base,
    sources,
    vocab_size,
    *,
    max_macro_length,
    max_bytes_per_source,
    min_frequency,
    word_bounded,
    merges_per_round,
):
    base_vocab = base.get_vocab_size()
    boundaries = boundary_base_ids(base)
    pieces, starts, doc_starts, total = [], [], [], 0
    for encoding in _iter_base_encodings(base, sources, max_bytes_per_source):
        doc_starts.append(total)
        pieces.append(np.asarray(encoding.ids, dtype=np.int32))
        if word_bounded:
            starts.append(word_starts_from_word_ids(encoding.word_ids))
        total += len(encoding.ids)
    x = np.concatenate(pieces).astype(np.int64)
    del pieces
    word_starts = np.concatenate(starts) if word_bounded else None
    del starts
    empty = MacroTokenizer(
        base,
        [],
        max_macro_length=max_macro_length,
        boundary_ids=boundaries,
        segmentation="greedy",
        word_bounded=word_bounded,
    )
    stops = empty.greedy_stops(x, np.asarray(doc_starts, dtype=np.int64), word_starts)
    del word_starts
    n = len(x)
    merges: list[tuple[int, int]] = []
    while base_vocab + len(merges) < vocab_size:
        if merges:
            pairs = np.asarray(merges, dtype=np.int64)
            keys = pairs[:, 0] * base_vocab + pairs[:, 1]
            order = np.argsort(keys, kind="stable")
            edge_keys = keys[order]
            edge_children = base_vocab + order.astype(np.int64)
        else:
            edge_keys = edge_children = np.zeros(0, dtype=np.int64)
        positions, macro_ids, widths = greedy_parse(
            x, stops, edge_keys, edge_children, base_vocab, max_macro_length
        )
        following = positions + widths
        usable = (following < n) & (widths < max_macro_length)
        usable[usable] &= ~stops[following[usable]]
        candidates = macro_ids[usable] * base_vocab + x[following[usable]]
        del positions, macro_ids, widths, following, usable
        keys, counts = np.unique(candidates, return_counts=True)
        del candidates
        keep = counts >= min_frequency
        keys, counts = keys[keep], counts[keep]
        if not len(keys):
            break
        room = vocab_size - base_vocab - len(merges)
        best = np.lexsort((keys, -counts))[: min(merges_per_round, room)]
        for key in keys[best].tolist():
            parent, token = divmod(key, base_vocab)
            merges.append((parent, token))
    return MacroTokenizer(
        base,
        merges,
        max_macro_length=max_macro_length,
        boundary_ids=boundaries,
        segmentation="greedy",
        word_bounded=word_bounded,
    )

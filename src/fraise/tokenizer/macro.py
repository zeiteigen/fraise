"""A macro vocabulary learned as merges over sequences of base-token IDs.

Text is first encoded by the base tokenizer. Each base ID is then mapped to one
private-use code point and a second stage groups those code points into macro
tokens. Every macro therefore decomposes deterministically into base IDs, and
concatenating the decompositions of an encoding reproduces the base encoding
exactly.

Macro IDs ``0 .. base_vocab_size - 1`` are the base tokens themselves; merge
``k`` creates macro ID ``base_vocab_size + k``. Base special tokens never merge,
and macros never continue past a base token that contains a line break.

Two segmentations are supported:

``bpe`` (the default) replays the merges in rank order, like any BPE. The first
``word_bounded_merges`` merges apply only inside a word (one pre-token of the
base tokenizer); the rest apply across words. ``word_bounded_merges=0`` is the
original hierarchical tokenizer; ``len(merges)`` keeps every macro inside a
word; anything between is a two-stage curriculum in the manner of SuperBPE.

``greedy`` requires every merge to append one base token to an existing entry,
so the vocabulary is prefix-closed, and segments left to right by longest
match. Whether a macro ends before base token ``t`` then depends only on the
macro so far and on token ``t`` itself: no boundary waits on later text.
With ``word_bounded=True`` the longest match never extends across a word start.
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split

FORMAT = "fraise-macro-tokenizer"
FORMAT_VERSION = 1
# Version 2 adds the segmentation fields; version 1 files are plain cross-word BPE.
SEGMENTED_FORMAT_VERSION = 2
SEGMENTATIONS = ("bpe", "greedy")
# Supplementary Private Use Area-A: one code point per base ID, never normalized.
_CHAR_OFFSET = 0xF0000
MAX_BASE_VOCAB_SIZE = 0xFFFFE - _CHAR_OFFSET
# Marks a word start for the word-bounded BPE stage; never a macro character.
_WORD_SEPARATOR = "\U0010fffd"


def base_ids_to_text(ids: Sequence[int]) -> str:
    """Map base IDs to one private-use character each."""
    values = np.asarray(ids, dtype=np.uint32) + _CHAR_OFFSET
    return values.astype("<u4").tobytes().decode("utf-32-le")


def text_to_base_ids(text: str) -> list[int]:
    values = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
    return (values.astype(np.int64) - _CHAR_OFFSET).tolist()


def _word_separated_text(ids: Sequence[int], word_starts: np.ndarray) -> str:
    """Base IDs as characters with a separator before every word start but the first."""
    values = np.asarray(ids, dtype=np.int64) + _CHAR_OFFSET
    starts = np.flatnonzero(np.asarray(word_starts, dtype=bool)[1:]) + 1
    values = np.insert(values, starts, ord(_WORD_SEPARATOR))
    return values.astype("<u4").tobytes().decode("utf-32-le")


def word_starts_from_word_ids(word_ids: Sequence[int | None]) -> np.ndarray:
    """True where a base token starts a new pre-token (word) of the base tokenizer."""
    starts = np.ones(len(word_ids), dtype=bool)
    for index in range(1, len(word_ids)):
        current = word_ids[index]
        starts[index] = current is None or current != word_ids[index - 1]
    return starts


@dataclass
class MacroEncoding:
    """Macro IDs; single-text encodes also carry display tokens and offsets."""

    ids: list[int]
    base_ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    offsets: list[tuple[int, int]] = field(default_factory=list)


def special_base_ids(base: Tokenizer) -> list[int]:
    """Added tokens plus Fraise's required markers; none of them ever merge."""
    from .loading import SPECIAL_TOKENS

    required = {base.token_to_id(token) for token in SPECIAL_TOKENS}
    return sorted((set(base.get_added_tokens_decoder()) | required) - {None})


def boundary_base_ids(base: Tokenizer) -> list[int]:
    """Non-special base tokens that contain a line break end a macro segment."""
    specials = set(special_base_ids(base))
    return [
        token_id
        for token_id in range(base.get_vocab_size())
        if token_id not in specials
        and "\n" in base.decode([token_id], skip_special_tokens=False)
    ]


def _segmenting_tokenizer(
    base: Tokenizer, boundary_ids: Sequence[int], *, word_bounded: bool = False
) -> Tokenizer:
    """Configure added specials and boundaries on an empty macro-stage BPE."""
    tokenizer = Tokenizer(BPE())
    _configure_macro_stage(
        tokenizer,
        special_base_ids(base),
        base_ids_to_text(boundary_ids),
        word_bounded=word_bounded,
    )
    return tokenizer


def _configure_macro_stage(tokenizer, specials, boundary_characters, *, word_bounded):
    splits = []
    if word_bounded:
        splits.append(Split(Regex(_WORD_SEPARATOR), behavior="removed"))
    if boundary_characters:
        splits.append(
            Split(Regex(f"[{boundary_characters}]"), behavior="merged_with_previous")
        )
    if len(splits) == 1:
        tokenizer.pre_tokenizer = splits[0]
    elif splits:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(splits)
    tokenizer.add_special_tokens(
        [
            AddedToken(base_ids_to_text([token_id]), special=True, normalized=False)
            for token_id in specials
        ]
    )


def greedy_parse(
    x: np.ndarray,
    stops: np.ndarray,
    edge_keys: np.ndarray,
    edge_children: np.ndarray,
    base_vocab_size: int,
    max_macro_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longest-match segmentation of a base-ID array over a prefix-closed trie.

    ``stops[i]`` forbids a macro from extending into position ``i`` (a sequence
    start, a special token or the token after one, the token after a line break,
    a word start when word-bounded). Returns the start position of every macro,
    its ID and its length in base tokens, in order.
    """
    n = len(x)
    node = x.astype(np.int64, copy=True)
    length = np.ones(n, dtype=np.int64)
    active = np.flatnonzero(~stops[1:] if n > 1 else np.zeros(0, dtype=bool))
    # ``active`` holds positions whose macro may still grow by one more token.
    for grow in range(1, max_macro_length):
        if not active.size or not edge_keys.size:
            break
        nxt = active + grow
        keep = nxt < n
        active, nxt = active[keep], nxt[keep]
        keep = ~stops[nxt]
        active, nxt = active[keep], nxt[keep]
        keys = node[active] * base_vocab_size + x[nxt]
        where = np.minimum(np.searchsorted(edge_keys, keys), len(edge_keys) - 1)
        found = edge_keys[where] == keys
        active = active[found]
        node[active] = edge_children[where[found]]
        length[active] = grow + 1
    starts = stops.copy()
    if n:
        starts[0] = True
    on_path = np.zeros(n, dtype=bool)
    frontier = np.flatnonzero(starts)
    while frontier.size:
        on_path[frontier] = True
        frontier = frontier + length[frontier]
        frontier = frontier[frontier < n]
        frontier = frontier[~starts[frontier]]
    positions = np.flatnonzero(on_path)
    return positions, node[positions], length[positions]


class MacroTokenizer:
    """Duck-types the parts of ``tokenizers.Tokenizer`` that Fraise uses."""

    def __init__(
        self,
        base: Tokenizer,
        merges: Sequence[tuple[int, int]],
        *,
        max_macro_length: int,
        boundary_ids: Sequence[int],
        segmentation: str = "bpe",
        word_bounded_merges: int = 0,
        word_bounded: bool = False,
    ):
        base_vocab = base.get_vocab_size()
        if base_vocab > MAX_BASE_VOCAB_SIZE:
            raise ValueError(f"base vocabulary cannot exceed {MAX_BASE_VOCAB_SIZE}")
        if segmentation not in SEGMENTATIONS:
            raise ValueError(f"segmentation must be one of {SEGMENTATIONS}")
        self.base = base
        self.base_vocab_size = base_vocab
        self.merges = [(int(left), int(right)) for left, right in merges]
        self.boundary_ids = sorted(int(token_id) for token_id in boundary_ids)
        self.configured_max_macro_length = int(max_macro_length)
        self.segmentation = segmentation
        self.word_bounded_merges = int(word_bounded_merges)
        if segmentation == "bpe":
            if word_bounded:
                raise ValueError("bpe segmentation bounds words with word_bounded_merges")
            if not 0 <= self.word_bounded_merges <= len(self.merges):
                raise ValueError("word_bounded_merges must be between 0 and the merges")
            self.word_bounded = self.word_bounded_merges > 0
        else:
            if self.word_bounded_merges:
                raise ValueError("greedy segmentation bounds words with word_bounded")
            self.word_bounded = bool(word_bounded)

        decompositions: list[tuple[int, ...]] = [(i,) for i in range(base_vocab)]
        for index, (left, right) in enumerate(self.merges):
            if not (0 <= left < base_vocab + index and 0 <= right < base_vocab + index):
                raise ValueError(f"merge {index} refers to an undefined macro")
            if segmentation == "greedy" and right >= base_vocab:
                raise ValueError("greedy segmentation needs merges that append one base token")
            decompositions.append(decompositions[left] + decompositions[right])
        self._decompositions = decompositions
        lengths = np.fromiter(map(len, decompositions), dtype=np.int64)
        self.unit_lengths = lengths
        self.max_macro_length = int(lengths.max())
        if self.max_macro_length > self.configured_max_macro_length:
            raise ValueError("a macro exceeds the configured max_macro_length")
        table = np.zeros((len(decompositions), self.max_macro_length), dtype=np.int64)
        for macro_id, parts in enumerate(decompositions):
            table[macro_id, : len(parts)] = parts
        self.decompositions = table

        vocab = {base_ids_to_text(parts): i for i, parts in enumerate(decompositions)}
        if len(vocab) != len(decompositions):
            raise ValueError("macro merges must create distinct tokens")
        specials = special_base_ids(base)
        self._special = np.zeros(base_vocab, dtype=bool)
        self._special[specials] = True
        self._line_break = np.zeros(base_vocab, dtype=bool)
        self._line_break[self.boundary_ids] = True
        self._stage1 = self._stage2 = None
        if segmentation == "greedy":
            keys = np.array(
                [left * base_vocab + right for left, right in self.merges], dtype=np.int64
            )
            order = np.argsort(keys, kind="stable")
            self._edge_keys = keys[order]
            self._edge_children = (base_vocab + np.arange(len(keys), dtype=np.int64))[order]
        else:
            self._build_bpe_stages(specials)
        self._serialized = None

    def _build_bpe_stages(self, specials):
        base_vocab, split = self.base_vocab_size, self.word_bounded_merges
        decompositions = self._decompositions
        if split:
            stage1 = Tokenizer(
                BPE(
                    vocab={
                        base_ids_to_text(decompositions[i]): i
                        for i in range(base_vocab + split)
                    },
                    merges=[
                        (
                            base_ids_to_text(decompositions[left]),
                            base_ids_to_text(decompositions[right]),
                        )
                        for left, right in self.merges[:split]
                    ],
                )
            )
            _configure_macro_stage(
                stage1, specials, base_ids_to_text(self.boundary_ids), word_bounded=True
            )
            self._stage1 = stage1
        if split == len(self.merges) and split:
            return
        # The cross-word stage reads stage-one IDs, each as one character.
        alphabet = base_vocab + split
        if alphabet > MAX_BASE_VOCAB_SIZE:
            raise ValueError("the word-bounded stage leaves too many symbols")
        strings = [base_ids_to_text([i]) for i in range(alphabet)]
        for left, right in self.merges[split:]:
            strings.append(strings[left] + strings[right])
        stage2 = Tokenizer(
            BPE(
                vocab={string: i for i, string in enumerate(strings)},
                merges=[(strings[left], strings[right]) for left, right in self.merges[split:]],
            )
        )
        line_breaks = set(self.boundary_ids)
        ends_line = [
            i for i in range(alphabet) if decompositions[i][-1] in line_breaks
        ]
        _configure_macro_stage(
            stage2, specials, base_ids_to_text(ends_line), word_bounded=False
        )
        self._stage2 = stage2

    # Tokenizer protocol -------------------------------------------------

    def get_vocab_size(self) -> int:
        return len(self._decompositions)

    def token_to_id(self, token: str) -> int | None:
        return self.base.token_to_id(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> MacroEncoding:
        base = self.base.encode(text, add_special_tokens=add_special_tokens)
        starts = word_starts_from_word_ids(base.word_ids) if self.word_bounded else None
        ids = self._macro_ids([base.ids], [starts])[0]
        tokens, offsets, cursor = [], [], 0
        for macro_id in ids:
            width = int(self.unit_lengths[macro_id])
            spans = [
                span
                for span in base.offsets[cursor : cursor + width]
                if span[1] > span[0]
            ]
            offsets.append((spans[0][0], spans[-1][1]) if spans else (0, 0))
            tokens.append(self.decode([macro_id], skip_special_tokens=False))
            cursor += width
        return MacroEncoding(ids, list(base.ids), tokens, offsets)

    def encode_batch(
        self, texts: Iterable[str], add_special_tokens: bool = True
    ) -> list[MacroEncoding]:
        encodings = self.base.encode_batch(
            list(texts), add_special_tokens=add_special_tokens
        )
        starts = [
            word_starts_from_word_ids(encoding.word_ids) if self.word_bounded else None
            for encoding in encodings
        ]
        return [
            MacroEncoding(ids)
            for ids in self._macro_ids([encoding.ids for encoding in encodings], starts)
        ]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.base.decode(
            self.expand(ids), skip_special_tokens=skip_special_tokens
        )

    def to_str(self) -> str:
        if self._serialized is None:
            values = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "base_tokenizer_sha256": hashlib.sha256(
                    self.base.to_str().encode("utf-8")
                ).hexdigest(),
                "max_macro_length": self.configured_max_macro_length,
                "boundary_tokens": self.boundary_ids,
            }
            if self.segmentation != "bpe" or self.word_bounded_merges:
                values["format_version"] = SEGMENTED_FORMAT_VERSION
                values["segmentation"] = self.segmentation
                if self.segmentation == "bpe":
                    values["word_bounded_merges"] = self.word_bounded_merges
                else:
                    values["word_bounded"] = self.word_bounded
            values["merges"] = self.merges
            values["base_tokenizer"] = self.base.to_str()
            self._serialized = json.dumps(values, separators=(",", ":"))
        return self._serialized

    # Macro structure ----------------------------------------------------

    def expand(self, ids: Iterable[int]) -> list[int]:
        """Concatenate the base decompositions of macro IDs."""
        decompositions = self._decompositions
        return [base_id for macro_id in ids for base_id in decompositions[macro_id]]

    def decomposition(self, macro_id: int) -> tuple[int, ...]:
        return self._decompositions[macro_id]

    def encode_base_ids(
        self, base_ids: Sequence[int], word_starts: np.ndarray | None = None
    ) -> list[int]:
        """Apply the macro segmentation to an existing base-ID sequence.

        Word-bounded segmentations need the base tokenizer's word starts; when
        they are not given they are recovered from the base-ID strings.
        """
        if self.word_bounded and word_starts is None:
            word_starts = self.word_starts(base_ids)
        return self._macro_ids([base_ids], [word_starts])[0]

    def word_starts(self, base_ids: Sequence[int]) -> np.ndarray:
        """Where each base token starts a pre-token, found by re-running the
        base pre-tokenizer over the tokens' byte-level strings."""
        ids = list(base_ids)
        starts = np.ones(len(ids), dtype=bool)
        run: list[int] = []
        for index, token_id in enumerate([*ids, None]):
            if token_id is not None and not self._special[token_id]:
                run.append(index)
                continue
            if run:
                pieces = [self.base.id_to_token(ids[i]) for i in run]
                text = self.base.decode([ids[i] for i in run], skip_special_tokens=False)
                cuts, position = set(), 0
                for piece, _ in self.base.pre_tokenizer.pre_tokenize_str(text):
                    cuts.add(position)
                    position += len(piece)
                position = 0
                for i, piece in zip(run, pieces):
                    starts[i] = position in cuts
                    position += len(piece)
                run = []
        return starts

    def _macro_ids(
        self, sequences: list[Sequence[int]], word_starts: list[np.ndarray | None]
    ) -> list[list[int]]:
        if self.segmentation == "greedy":
            return self._greedy_ids(sequences, word_starts)
        ids = sequences
        if self._stage1 is not None:
            texts = [
                _word_separated_text(sequence, starts)
                for sequence, starts in zip(ids, word_starts)
            ]
            ids = [
                encoding.ids
                for encoding in self._stage1.encode_batch(texts, add_special_tokens=False)
            ]
        if self._stage2 is not None:
            texts = [base_ids_to_text(sequence) for sequence in ids]
            ids = [
                encoding.ids
                for encoding in self._stage2.encode_batch(texts, add_special_tokens=False)
            ]
        return [list(sequence) for sequence in ids]

    def greedy_stops(
        self, x: np.ndarray, sequence_starts: np.ndarray, word_starts: np.ndarray | None
    ) -> np.ndarray:
        """Positions no macro may extend into (see ``greedy_parse``)."""
        stops = np.zeros(len(x), dtype=bool)
        stops[sequence_starts] = True
        special = self._special[x]
        stops |= special
        stops[1:] |= special[:-1] | self._line_break[x[:-1]]
        if self.word_bounded and word_starts is not None:
            stops |= word_starts
        return stops

    def _greedy_ids(self, sequences, word_starts):
        lengths = np.fromiter(map(len, sequences), dtype=np.int64, count=len(sequences))
        if not lengths.sum():
            return [[] for _ in sequences]
        x = np.concatenate([np.asarray(s, dtype=np.int64) for s in sequences])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        starts = None
        if self.word_bounded:
            starts = np.concatenate(
                [
                    np.asarray(s, dtype=bool)
                    if s is not None
                    else self.word_starts(sequence)
                    for s, sequence in zip(word_starts, sequences)
                ]
            )
        stops = self.greedy_stops(x, offsets[:-1][lengths > 0], starts)
        positions, ids, _ = greedy_parse(
            x,
            stops,
            self._edge_keys,
            self._edge_children,
            self.base_vocab_size,
            self.max_macro_length,
        )
        bounds = np.searchsorted(positions, offsets)
        ids = ids.tolist()
        return [ids[bounds[i] : bounds[i + 1]] for i in range(len(sequences))]

    @classmethod
    def from_str(cls, contents: str) -> MacroTokenizer:
        values = json.loads(contents)
        if values.get("format") != FORMAT:
            raise ValueError("not a Fraise macro tokenizer")
        version = values.get("format_version")
        if version not in (FORMAT_VERSION, SEGMENTED_FORMAT_VERSION):
            raise ValueError(f"unsupported macro tokenizer version {version!r}")
        base_contents = values["base_tokenizer"]
        if (
            hashlib.sha256(base_contents.encode("utf-8")).hexdigest()
            != values["base_tokenizer_sha256"]
        ):
            raise ValueError("macro tokenizer's base tokenizer checksum does not match")
        options = {}
        if version == SEGMENTED_FORMAT_VERSION:
            options["segmentation"] = values["segmentation"]
            if values["segmentation"] == "bpe":
                options["word_bounded_merges"] = values["word_bounded_merges"]
            else:
                options["word_bounded"] = values["word_bounded"]
        tokenizer = cls(
            Tokenizer.from_str(base_contents),
            [tuple(pair) for pair in values["merges"]],
            max_macro_length=values["max_macro_length"],
            boundary_ids=values["boundary_tokens"],
            **options,
        )
        tokenizer._serialized = contents
        return tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> MacroTokenizer:
        return cls.from_str(Path(path).expanduser().read_text(encoding="utf-8"))


def is_macro_tokenizer_text(contents: str) -> bool:
    # The format marker is the first serialized key; avoid parsing large files twice.
    return contents.lstrip().startswith('{"format":"' + FORMAT + '"')


def unit_lengths(tokenizer) -> np.ndarray | None:
    """Base tokens per ID for macro tokenizers; ``None`` when every ID is one unit."""
    return getattr(tokenizer, "unit_lengths", None)


def counting_tokenizer(tokenizer):
    """The tokenizer whose IDs measure training budgets (the shared base level)."""
    return getattr(tokenizer, "base", tokenizer)

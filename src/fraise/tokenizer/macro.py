"""A macro vocabulary learned as merges over sequences of base-token IDs.

Text is first encoded by the base tokenizer. Each base ID is then mapped to one
private-use code point and a second BPE stage merges those code points into
macro tokens. Every macro therefore decomposes deterministically into base IDs,
and concatenating the decompositions of an encoding reproduces the base
encoding exactly.

Macro IDs ``0 .. base_vocab_size - 1`` are the base tokens themselves; merge
``k`` creates macro ID ``base_vocab_size + k``. Base special tokens never merge,
and macros never continue past a base token that contains a line break.
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split

FORMAT = "fraise-macro-tokenizer"
FORMAT_VERSION = 1
# Supplementary Private Use Area-A: one code point per base ID, never normalized.
_CHAR_OFFSET = 0xF0000
MAX_BASE_VOCAB_SIZE = 0xFFFFE - _CHAR_OFFSET


def base_ids_to_text(ids: Sequence[int]) -> str:
    """Map base IDs to one private-use character each."""
    values = np.asarray(ids, dtype=np.uint32) + _CHAR_OFFSET
    return values.astype("<u4").tobytes().decode("utf-32-le")


def text_to_base_ids(text: str) -> list[int]:
    values = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
    return (values.astype(np.int64) - _CHAR_OFFSET).tolist()


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


def _segmenting_tokenizer(base: Tokenizer, boundary_ids: Sequence[int]) -> Tokenizer:
    """Configure added specials and boundaries on an empty macro-stage BPE."""
    tokenizer = Tokenizer(BPE())
    _configure_macro_stage(tokenizer, base, boundary_ids)
    return tokenizer


def _configure_macro_stage(tokenizer, base, boundary_ids):
    if boundary_ids:
        characters = "".join(base_ids_to_text(boundary_ids))
        tokenizer.pre_tokenizer = Split(
            Regex(f"[{characters}]"), behavior="merged_with_previous"
        )
    tokenizer.add_special_tokens(
        [
            AddedToken(base_ids_to_text([token_id]), special=True, normalized=False)
            for token_id in special_base_ids(base)
        ]
    )


class MacroTokenizer:
    """Duck-types the parts of ``tokenizers.Tokenizer`` that Fraise uses."""

    def __init__(
        self,
        base: Tokenizer,
        merges: Sequence[tuple[int, int]],
        *,
        max_macro_length: int,
        boundary_ids: Sequence[int],
    ):
        base_vocab = base.get_vocab_size()
        if base_vocab > MAX_BASE_VOCAB_SIZE:
            raise ValueError(f"base vocabulary cannot exceed {MAX_BASE_VOCAB_SIZE}")
        self.base = base
        self.base_vocab_size = base_vocab
        self.merges = [(int(left), int(right)) for left, right in merges]
        self.boundary_ids = sorted(int(token_id) for token_id in boundary_ids)
        self.configured_max_macro_length = int(max_macro_length)

        decompositions: list[tuple[int, ...]] = [(i,) for i in range(base_vocab)]
        for index, (left, right) in enumerate(self.merges):
            if not (0 <= left < base_vocab + index and 0 <= right < base_vocab + index):
                raise ValueError(f"merge {index} refers to an undefined macro")
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
        text_merges = [
            (
                base_ids_to_text(decompositions[left]),
                base_ids_to_text(decompositions[right]),
            )
            for left, right in self.merges
        ]
        self._macro = Tokenizer(BPE(vocab=vocab, merges=text_merges))
        _configure_macro_stage(self._macro, base, self.boundary_ids)
        self._serialized = None

    # Tokenizer protocol -------------------------------------------------

    def get_vocab_size(self) -> int:
        return len(self._decompositions)

    def token_to_id(self, token: str) -> int | None:
        return self.base.token_to_id(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> MacroEncoding:
        base = self.base.encode(text, add_special_tokens=add_special_tokens)
        ids = self._macro_ids([base.ids])[0]
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
        return [
            MacroEncoding(ids)
            for ids in self._macro_ids([encoding.ids for encoding in encodings])
        ]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.base.decode(
            self.expand(ids), skip_special_tokens=skip_special_tokens
        )

    def to_str(self) -> str:
        if self._serialized is None:
            self._serialized = json.dumps(
                {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "base_tokenizer_sha256": hashlib.sha256(
                        self.base.to_str().encode("utf-8")
                    ).hexdigest(),
                    "max_macro_length": self.configured_max_macro_length,
                    "boundary_tokens": self.boundary_ids,
                    "merges": self.merges,
                    "base_tokenizer": self.base.to_str(),
                },
                separators=(",", ":"),
            )
        return self._serialized

    # Macro structure ----------------------------------------------------

    def expand(self, ids: Iterable[int]) -> list[int]:
        """Concatenate the base decompositions of macro IDs."""
        decompositions = self._decompositions
        return [base_id for macro_id in ids for base_id in decompositions[macro_id]]

    def decomposition(self, macro_id: int) -> tuple[int, ...]:
        return self._decompositions[macro_id]

    def encode_base_ids(self, base_ids: Sequence[int]) -> list[int]:
        """Apply the macro merges to an existing base-ID sequence."""
        return self._macro_ids([base_ids])[0]

    def _macro_ids(self, sequences: list[Sequence[int]]) -> list[list[int]]:
        texts = [base_ids_to_text(ids) for ids in sequences]
        return [
            encoding.ids
            for encoding in self._macro.encode_batch(texts, add_special_tokens=False)
        ]

    @classmethod
    def from_str(cls, contents: str) -> MacroTokenizer:
        values = json.loads(contents)
        if values.get("format") != FORMAT:
            raise ValueError("not a Fraise macro tokenizer")
        if values.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported macro tokenizer version {values.get('format_version')!r}"
            )
        base_contents = values["base_tokenizer"]
        if (
            hashlib.sha256(base_contents.encode("utf-8")).hexdigest()
            != values["base_tokenizer_sha256"]
        ):
            raise ValueError("macro tokenizer's base tokenizer checksum does not match")
        tokenizer = cls(
            Tokenizer.from_str(base_contents),
            [tuple(pair) for pair in values["merges"]],
            max_macro_length=values["max_macro_length"],
            boundary_ids=values["boundary_tokens"],
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

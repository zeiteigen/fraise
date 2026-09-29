"""Load tokenizers without mutation and identify required special tokens."""

import hashlib
from pathlib import Path

from tokenizers import Tokenizer

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>")


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def load_tokenizer(path: str | Path) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(Path(path).expanduser()))
    for token in SPECIAL_TOKENS:
        special_token_id(tokenizer, token)
    return tokenizer

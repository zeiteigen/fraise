"""Load tokenizers without mutation and identify required special tokens."""

import hashlib
from pathlib import Path

from tokenizers import Tokenizer

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>")


def tokenizer_fingerprint(tokenizer) -> str:
    return hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()


def special_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def load_tokenizer(path: str | Path):
    """Load a base tokenizer, or a macro tokenizer with its embedded base."""
    from .macro import MacroTokenizer, is_macro_tokenizer_text

    path = Path(path).expanduser()
    contents = path.read_text(encoding="utf-8")
    if is_macro_tokenizer_text(contents):
        tokenizer = MacroTokenizer.from_str(contents)
    else:
        tokenizer = Tokenizer.from_str(contents)
    for token in SPECIAL_TOKENS:
        special_token_id(tokenizer, token)
    return tokenizer

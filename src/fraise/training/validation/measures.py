"""Per-token units and byte lengths that make held-out losses comparable.

Token-level losses depend on the tokenizer, so validation also reports bits per
byte (BPB): summed negative log-likelihood in bits divided by the UTF-8 bytes
the scored targets cover. Byte lengths are exact for byte-level BPE vocabularies
(each vocabulary character stands for one byte); other tokenizers report no BPB.
"""

import json
from dataclasses import dataclass

import numpy as np
from tokenizers.pre_tokenizers import ByteLevel

from fraise.tokenizer.macro import counting_tokenizer, special_base_ids, unit_lengths


@dataclass(frozen=True)
class ValidationMeasures:
    unit_lengths: np.ndarray | None = None
    byte_lengths: np.ndarray | None = None


def base_byte_lengths(tokenizer) -> np.ndarray | None:
    """UTF-8 bytes per ID of a byte-level BPE tokenizer; specials have none."""
    decoder = json.loads(tokenizer.to_str()).get("decoder") or {}
    if decoder.get("type") != "ByteLevel":
        return None
    specials = set(special_base_ids(tokenizer))
    alphabet = set(ByteLevel.alphabet())
    lengths = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in specials:
            continue
        token = tokenizer.id_to_token(token_id)
        if token is None or not set(token) <= alphabet:
            return None
        lengths[token_id] = len(token)
    return lengths


def validation_measures(tokenizer) -> ValidationMeasures:
    units = unit_lengths(tokenizer)
    byte_lengths = base_byte_lengths(counting_tokenizer(tokenizer))
    table = getattr(tokenizer, "decompositions", None)
    if byte_lengths is not None and table is not None:
        width = np.arange(table.shape[1])
        present = width[None, :] < units[:, None]
        byte_lengths = (byte_lengths[table] * present).sum(axis=1)
    return ValidationMeasures(units, byte_lengths)

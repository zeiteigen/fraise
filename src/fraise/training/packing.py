"""Pack token streams into shifted causal windows and count capacity."""

from collections import deque
from collections.abc import Iterator
from itertools import islice

import torch
from datasets import Dataset

from .batch_types import PackedTokenBatch
from .tokenization import TOKEN_IDS_COLUMN

_TOKEN_SEQUENCE_BATCH_SIZE = 1000


def _token_sequences(
    split: Dataset,
) -> Iterator[list[int]]:
    """Yield pre-tokenized document sequences from a dataset split."""
    if TOKEN_IDS_COLUMN not in split.column_names:
        raise ValueError(
            f"training datasets must contain a '{TOKEN_IDS_COLUMN}' column; "
            f"available columns: {split.column_names}"
        )

    for batch in split.select_columns(TOKEN_IDS_COLUMN).iter(
        batch_size=_TOKEN_SEQUENCE_BATCH_SIZE
    ):
        for token_ids in batch[TOKEN_IDS_COLUMN]:
            if token_ids:
                yield token_ids


def packed_target_capacity(
    split: Dataset,
    sequence_length: int,
    packing_stride: int,
) -> int:
    """Count target positions one epoch can produce from a tokenized split.

    Arrow list offsets give the concatenated stream length without loading
    every token ID into Python. This matches ``iter_packed_blocks`` including
    repeated targets when windows overlap.
    """
    if TOKEN_IDS_COLUMN not in split.column_names:
        raise ValueError(
            f"training datasets must contain a '{TOKEN_IDS_COLUMN}' column"
        )
    if sequence_length < 2 or packing_stride < 1 or packing_stride > sequence_length:
        raise ValueError("invalid sequence_length or packing_stride")

    token_count = 0
    for chunk in split.data.column(TOKEN_IDS_COLUMN).chunks:
        offsets = chunk.offsets
        token_count += offsets[-1].as_py() - offsets[0].as_py()

    if token_count < 2:
        return 0
    if token_count <= sequence_length:
        return token_count - 1

    full_blocks = (token_count - sequence_length - 1) // packing_stride + 1
    tail_tokens = token_count - full_blocks * packing_stride
    return full_blocks * sequence_length + max(tail_tokens - 1, 0)


def _make_block(
    token_ids: list[int],
    sequence_length: int,
    pad_token_id: int,
    valid_targets: int,
) -> PackedTokenBatch:
    """Turn one token window into shifted inputs, targets, and a target mask."""
    block = token_ids[: valid_targets + 1]
    block.extend([pad_token_id] * (sequence_length + 1 - len(block)))

    target_mask = torch.zeros(sequence_length, dtype=torch.bool)
    target_mask[:valid_targets] = True
    return PackedTokenBatch(
        input_ids=torch.tensor(block[:-1], dtype=torch.long),
        targets=torch.tensor(block[1:], dtype=torch.long),
        target_mask=target_mask,
    )


def iter_packed_blocks(
    split: Dataset,
    sequence_length: int,
    packing_stride: int,
    pad_token_id: int,
    *,
    seed: int,
    shuffle: bool,
    max_tokens: int | None,
) -> Iterator[PackedTokenBatch]:
    """Yield fixed-context causal-LM windows from a Hugging Face dataset.

    Documents are concatenated into one causal stream. BOS/EOS tokens mark
    document boundaries, but they do not prevent attention across adjacent
    documents because Fraise's model uses ordinary causal attention.
    """
    if shuffle:
        split = split.shuffle(seed=seed)

    # A deque avoids shifting every remaining token when an overlapping window
    # advances. The stream is intentionally continuous across documents.
    buffer: deque[int] = deque()
    produced_tokens = 0

    for sequence in _token_sequences(split):
        buffer.extend(sequence)

        while len(buffer) >= sequence_length + 1:
            remaining = (
                max_tokens - produced_tokens
                if max_tokens is not None
                else sequence_length
            )
            if remaining <= 0:
                return

            valid_targets = min(sequence_length, remaining)
            window = list(islice(buffer, valid_targets + 1))
            yield _make_block(
                window,
                sequence_length,
                pad_token_id,
                valid_targets,
            )
            produced_tokens += valid_targets

            if valid_targets < sequence_length:
                return
            for _ in range(packing_stride):
                buffer.popleft()

    if max_tokens is not None and produced_tokens >= max_tokens:
        return
    if len(buffer) < 2:
        return

    remaining = (
        max_tokens - produced_tokens if max_tokens is not None else sequence_length
    )
    valid_targets = min(sequence_length, len(buffer) - 1, remaining)
    if valid_targets > 0:
        yield _make_block(
            list(islice(buffer, valid_targets + 1)),
            sequence_length,
            pad_token_id,
            valid_targets,
        )

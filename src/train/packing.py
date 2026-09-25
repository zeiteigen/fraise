"""Pre-tokenize Fraise datasets and pack them into causal-LM batches."""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice

import torch
from datasets import Dataset
from tokenizers import Tokenizer


TOKEN_IDS_COLUMN = "token_ids"
_TOKEN_SEQUENCE_BATCH_SIZE = 1_000


@dataclass
class PackedTokenBatch:
    """A batch of fixed-length inputs and the targets that should contribute."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    """Return a required token ID from the project's tokenizer."""
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token!r}")
    return token_id


def pretokenize_split(
    split: Dataset,
    tokenizer: Tokenizer,
    *,
    tokenization_batch_size: int,
) -> Dataset:
    """Encode every document in ``split`` before packing starts.

    The resulting dataset contains one ``token_ids`` list per input document.
    Empty documents keep an empty list so the row order remains unchanged,
    while the packer skips them just as it did for empty text values before
    pre-tokenization was introduced.
    """
    if "text" not in split.column_names:
        raise ValueError(
            "training datasets must contain a 'text' column; "
            f"available columns: {split.column_names}"
        )
    if tokenization_batch_size <= 0:
        raise ValueError(
            "tokenization_batch_size must be positive; "
            f"got {tokenization_batch_size}"
        )

    def encode_batch(batch: dict[str, list[object]]) -> dict[str, list[list[int]]]:
        texts = batch["text"]
        token_ids = [[] for _ in texts]
        non_empty = [
            (row_index, text)
            for row_index, text in enumerate(texts)
            if text
        ]
        if non_empty:
            encodings = tokenizer.encode_batch(
                [text for _, text in non_empty],
                add_special_tokens=True,
            )
            for (row_index, _), encoding in zip(non_empty, encodings):
                token_ids[row_index] = encoding.ids

        return {TOKEN_IDS_COLUMN: token_ids}

    return split.map(
        encode_batch,
        batched=True,
        batch_size=tokenization_batch_size,
        remove_columns=split.column_names,
        desc="Tokenizing dataset",
    )


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


def _stack(blocks: list[PackedTokenBatch]) -> PackedTokenBatch:
    return PackedTokenBatch(
        input_ids=torch.stack([block.input_ids for block in blocks]),
        targets=torch.stack([block.targets for block in blocks]),
        target_mask=torch.stack([block.target_mask for block in blocks]),
    )


def iter_packed_batches(
    split: Dataset,
    sequence_length: int,
    packing_stride: int,
    pad_token_id: int,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    max_tokens: int | None,
    start_batch_index: int = 0,
) -> Iterator[tuple[int, PackedTokenBatch]]:
    """Group packed windows into batches, optionally skipping a resume prefix."""
    blocks: list[PackedTokenBatch] = []
    batch_index = 0

    for block in iter_packed_blocks(
        split,
        sequence_length,
        packing_stride,
        pad_token_id,
        seed=seed,
        shuffle=shuffle,
        max_tokens=max_tokens,
    ):
        blocks.append(block)
        if len(blocks) < batch_size:
            continue

        if batch_index >= start_batch_index:
            yield batch_index, _stack(blocks)
        batch_index += 1
        blocks = []

    if blocks and batch_index >= start_batch_index:
        yield batch_index, _stack(blocks)

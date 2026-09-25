"""Pre-tokenize Fraise datasets and pack them into causal-LM batches."""

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
from itertools import islice

import torch
from datasets import Dataset, Features, List, Value
from tokenizers import Tokenizer

from datasets_mgr import DATASET_NAME_COLUMN


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
    max_tokens: int | None = None,
) -> Dataset:
    """Encode every document in ``split`` before packing starts.

    The resulting dataset contains one ``token_ids`` list per input document.
    Without a token budget, empty documents keep an empty list so row order
    stays unchanged. With ``max_tokens``, empty documents are skipped and only
    the prefix needed for that many causal targets is encoded. One additional
    token is retained as the final input for the shifted targets. The result
    is stored in an on-disk Arrow cache.
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
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 0
    ):
        raise ValueError(
            "max_tokens must be a non-negative integer or None; "
            f"got {max_tokens!r}"
        )

    if max_tokens == 0:
        return Dataset.from_dict({TOKEN_IDS_COLUMN: []})

    has_source_names = DATASET_NAME_COLUMN in split.column_names
    columns = ["text"]
    features = {TOKEN_IDS_COLUMN: List(Value("int64"))}
    if has_source_names:
        columns.append(DATASET_NAME_COLUMN)
        features[DATASET_NAME_COLUMN] = Value("string")

    def encoded_rows() -> Iterator[dict[str, object]]:
        selected_token_count = 0
        target_token_count = max_tokens + 1 if max_tokens is not None else None

        for batch in split.select_columns(columns).iter(
            batch_size=tokenization_batch_size
        ):
            texts = batch["text"]
            non_empty = [
                (row_index, text)
                for row_index, text in enumerate(texts)
                if text
            ]
            encodings = (
                tokenizer.encode_batch(
                    [text for _, text in non_empty],
                    add_special_tokens=True,
                )
                if non_empty
                else []
            )
            token_ids_by_row = {
                row_index: encoding.ids
                for (row_index, _), encoding in zip(non_empty, encodings)
            }

            for row_index in range(len(texts)):
                token_ids = token_ids_by_row.get(row_index, [])
                if target_token_count is not None:
                    remaining = target_token_count - selected_token_count
                    token_ids = token_ids[:remaining]
                    if not token_ids:
                        continue
                    selected_token_count += len(token_ids)

                row: dict[str, object] = {TOKEN_IDS_COLUMN: token_ids}
                if has_source_names:
                    row[DATASET_NAME_COLUMN] = batch[DATASET_NAME_COLUMN][row_index]
                yield row

                if (
                    target_token_count is not None
                    and selected_token_count >= target_token_count
                ):
                    return

    # Avoid hashing the generator closure: it captures the entire source
    # dataset, which can itself be larger than available RAM. A stable key
    # also lets resumed runs reuse the tokenized Arrow cache.
    fingerprint = None
    if isinstance(tokenizer, Tokenizer):
        digest = hashlib.sha256()
        digest.update(
            repr(
                (
                    "pretokenize-v1",
                    split._fingerprint,
                    tokenization_batch_size,
                    max_tokens,
                )
            ).encode("utf-8")
        )
        digest.update(tokenizer.to_str().encode("utf-8"))
        fingerprint = digest.hexdigest()

    # from_generator writes Arrow batches to disk and returns a memory-mapped
    # Dataset; neither the Python lists nor the Arrow table grow in RAM with
    # the length of the training prefix.
    return Dataset.from_generator(
        encoded_rows,
        features=Features(features),
        keep_in_memory=False,
        fingerprint=fingerprint,
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


def iter_dataset_packed_batches(
    splits: Mapping[str, Dataset],
    token_budgets: Mapping[str, int],
    sequence_length: int,
    packing_stride: int,
    pad_token_id: int,
    *,
    batch_size: int,
    start_batch_index: int = 0,
    seed: int = 0,
    shuffle: bool = False,
) -> Iterator[tuple[int, PackedTokenBatch]]:
    """Pack exact token budgets from named dataset prefixes into batches.

    Sources are traversed in their stored row/token order by default. When
    ``shuffle`` is enabled, each source is shuffled before its token budget
    is applied. The final block for a source is shortened at the token
    boundary, so integer rounding in the percentages does not overrun the
    requested budget.
    """
    unknown_budgets = set(token_budgets) - set(splits)
    if unknown_budgets:
        raise ValueError(
            "token budgets contain unknown dataset name(s): "
            f"{', '.join(sorted(unknown_budgets))}"
        )

    blocks: list[PackedTokenBatch] = []
    batch_index = 0

    for dataset_name, split in splits.items():
        budget = token_budgets.get(dataset_name, 0)
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError(
                f"token budget for {dataset_name!r} must be an integer; "
                f"got {budget!r}"
            )
        if budget < 0:
            raise ValueError(
                f"token budget for {dataset_name!r} must be non-negative; "
                f"got {budget!r}"
            )
        if budget == 0:
            continue

        for block in iter_packed_blocks(
            split,
            sequence_length,
            packing_stride,
            pad_token_id,
            seed=seed,
            shuffle=shuffle,
            max_tokens=budget,
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

"""Encode text prefixes into memory-mapped token-ID datasets."""

import hashlib
from collections.abc import Iterator

from datasets import Dataset, Features, List, Value
from tokenizers import Tokenizer

from fraise.tokenizer.loading import tokenizer_fingerprint

TOKEN_IDS_COLUMN = "token_ids"


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
            f"tokenization_batch_size must be positive; got {tokenization_batch_size}"
        )
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 0
    ):
        raise ValueError(
            f"max_tokens must be a non-negative integer or None; got {max_tokens!r}"
        )

    if max_tokens == 0:
        return Dataset.from_dict({TOKEN_IDS_COLUMN: []})

    columns = ["text"]
    features = {TOKEN_IDS_COLUMN: List(Value("int64"))}

    def encoded_rows() -> Iterator[dict[str, object]]:
        selected_token_count = 0
        target_token_count = max_tokens + 1 if max_tokens is not None else None

        for batch in split.select_columns(columns).iter(
            batch_size=tokenization_batch_size
        ):
            texts = batch["text"]
            non_empty = [
                (row_index, text) for row_index, text in enumerate(texts) if text
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
                yield row

                if (
                    target_token_count is not None
                    and selected_token_count >= target_token_count
                ):
                    return

    # Avoid hashing the generator closure: it captures the entire source
    # dataset, which can itself be larger than available RAM. A stable key
    # also lets resumed runs reuse the tokenized Arrow cache.
    fingerprint = hashlib.sha256(
        repr(
            (
                "pretokenize-v2",
                split._fingerprint,
                max_tokens,
                tokenizer_fingerprint(tokenizer),
            )
        ).encode()
    ).hexdigest()

    # from_generator writes Arrow batches to disk and returns a memory-mapped
    # Dataset; neither the Python lists nor the Arrow table grow in RAM with
    # the length of the training prefix.
    return Dataset.from_generator(
        encoded_rows,
        features=Features(features),
        keep_in_memory=False,
        fingerprint=fingerprint,
    )

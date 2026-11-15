"""Pack token streams into shifted causal windows and count capacity.

Documents are read straight from Arrow buffers as NumPy arrays; windows are
strided views of that stream. Window ``w`` always starts at stream offset
``w * stride``, so a resumed run can start at any window without packing
the windows before it.

Macro-token streams measure budgets in base-token *units*: a window's units are
the decomposition lengths of its target macros. Their final window keeps the
fewest targets that reach the budget, so planned budgets are exact unit counts.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset

from .batch_types import PackedTokenBatch
from .tokenization import TOKEN_IDS_COLUMN

# Upper bound on window elements materialized per gather (8 MiB of int64).
_GATHER_ELEMENTS = 1 << 20


def _token_column(split: Dataset) -> pa.ChunkedArray:
    if TOKEN_IDS_COLUMN not in split.column_names:
        raise ValueError(
            f"training datasets must contain a '{TOKEN_IDS_COLUMN}' column; "
            f"available columns: {split.column_names}"
        )
    if split._indices is not None:
        # Row selections/shuffles are index mappings over the Arrow table.
        split = split.flatten_indices()
    return split.data.column(TOKEN_IDS_COLUMN)


def _check_geometry(sequence_length: int, packing_stride: int) -> None:
    if sequence_length < 2 or packing_stride < 1 or packing_stride > sequence_length:
        raise ValueError("invalid sequence_length or packing_stride")


def _full_window_count(token_count: int, sequence_length: int, stride: int) -> int:
    if token_count < sequence_length + 1:
        return 0
    return (token_count - sequence_length - 1) // stride + 1


def packed_target_capacity(
    split: Dataset,
    sequence_length: int,
    packing_stride: int,
) -> int:
    """Count target positions a tokenized split can produce.

    Arrow list offsets give the concatenated stream length without loading
    every token ID. This matches ``iter_packed_blocks`` including repeated
    targets when windows overlap.
    """
    column = _token_column(split)
    _check_geometry(sequence_length, packing_stride)
    token_count = sum(
        chunk.offsets[-1].as_py() - chunk.offsets[0].as_py() for chunk in column.chunks
    )
    if token_count < 2:
        return 0
    full_blocks = _full_window_count(token_count, sequence_length, packing_stride)
    tail_tokens = token_count - full_blocks * packing_stride
    return full_blocks * sequence_length + min(max(tail_tokens - 1, 0), sequence_length)


class _TokenStream:
    """Random access to the concatenated token stream of a split, in document order."""

    def __init__(self, column: pa.ChunkedArray):
        lengths = pc.fill_null(pc.list_value_length(column), 0)
        lengths = lengths.to_numpy().astype(np.int64)
        self._column = column
        # Document d occupies stream positions [starts[d], starts[d + 1]).
        self._starts = np.concatenate(([0], np.cumsum(lengths)))
        self.length = int(self._starts[-1])

    def tokens(self, start: int, stop: int) -> np.ndarray:
        """Stream tokens in ``[start, stop)``."""
        first = int(np.searchsorted(self._starts, start, side="right")) - 1
        last = int(np.searchsorted(self._starts, stop, side="left"))
        documents = self._column.slice(first, last - first)
        values = pc.list_flatten(documents).to_numpy()
        offset = start - int(self._starts[first])
        return values[offset : offset + stop - start]


@dataclass(frozen=True)
class WindowPlan:
    """How many windows a budget needs and the exact units they produce."""

    windows: int
    units: int


def _stream(split: Dataset) -> _TokenStream:
    return _TokenStream(_token_column(split))


def _window_rows(stream, sequence_length, packing_stride, start_window=0):
    """Full windows from ``start_window`` as ``(first_index, rows)`` chunks."""
    full_windows = _full_window_count(stream.length, sequence_length, packing_stride)
    window_span = sequence_length + 1
    window = start_window
    while window < full_windows:
        count = min(full_windows - window, max(1, _GATHER_ELEMENTS // window_span))
        start = window * packing_stride
        tokens = stream.tokens(
            start, start + (count - 1) * packing_stride + window_span
        )
        rows = np.lib.stride_tricks.sliding_window_view(tokens, window_span)
        yield window, rows[::packing_stride]
        window += count


def _tail(stream, sequence_length, packing_stride) -> np.ndarray:
    """Tokens of the final partial window (inputs and targets), possibly empty."""
    full_windows = _full_window_count(stream.length, sequence_length, packing_stride)
    tail_start = full_windows * packing_stride
    if stream.length - tail_start < 2:
        return np.empty(0, dtype=np.int64)
    return stream.tokens(tail_start, stream.length)


def _targets_reaching(units: np.ndarray, needed: int) -> tuple[int, int]:
    """Fewest leading targets whose units reach ``needed``, and those units."""
    covered = np.cumsum(units)
    count = int(np.searchsorted(covered, needed)) + 1
    count = min(count, len(units))
    return count, int(covered[count - 1]) if count else 0


def plan_unit_windows(
    split: Dataset,
    sequence_length: int,
    packing_stride: int,
    budget: int | None,
    unit_lengths: np.ndarray,
) -> WindowPlan:
    """Scan a macro stream; with too little data ``units`` is the capacity."""
    _check_geometry(sequence_length, packing_stride)
    stream = _stream(split)
    budget = float("inf") if budget is None else budget
    produced = windows = 0
    for first, rows in _window_rows(stream, sequence_length, packing_stride):
        sums = unit_lengths[rows[:, 1:]].sum(axis=1)
        cumulative = produced + np.cumsum(sums)
        hit = int(np.searchsorted(cumulative, budget))
        if hit < len(rows):
            before = int(cumulative[hit - 1]) if hit else produced
            _, units = _targets_reaching(unit_lengths[rows[hit, 1:]], budget - before)
            return WindowPlan(first + hit + 1, before + units)
        produced, windows = int(cumulative[-1]), first + len(rows)
    tail = _tail(stream, sequence_length, packing_stride)
    if len(tail) and produced < budget:
        _, units = _targets_reaching(unit_lengths[tail[1:]], budget - produced)
        produced, windows = produced + units, windows + 1
    return WindowPlan(windows, produced)


def _units_before(stream, sequence_length, packing_stride, start_window, unit_lengths):
    produced = 0
    for first, rows in _window_rows(stream, sequence_length, packing_stride):
        rows = rows[: start_window - first]
        produced += int(unit_lengths[rows[:, 1:]].sum())
        if first + len(rows) >= start_window:
            break
    return produced


def _iter_unit_blocks(
    stream,
    sequence_length,
    packing_stride,
    pad_token_id,
    budget,
    start_window,
    unit_lengths,
):
    budget = float("inf") if budget is None else budget
    produced = _units_before(
        stream, sequence_length, packing_stride, start_window, unit_lengths
    )
    if produced >= budget:
        return
    full_mask = torch.ones(sequence_length, dtype=torch.bool)
    for _, rows in _window_rows(stream, sequence_length, packing_stride, start_window):
        units = unit_lengths[rows[:, 1:]]
        for row, row_units in zip(rows, units):
            total = int(row_units.sum())
            if produced + total >= budget:
                count, _ = _targets_reaching(row_units, budget - produced)
                yield _make_block(row, sequence_length, pad_token_id, count)
                return
            row = torch.from_numpy(row.astype(np.int64))
            yield PackedTokenBatch(row[:-1], row[1:], full_mask)
            produced += total
    full_windows = _full_window_count(stream.length, sequence_length, packing_stride)
    if start_window > full_windows or produced >= budget:
        return
    tail = _tail(stream, sequence_length, packing_stride)
    if len(tail):
        count, _ = _targets_reaching(unit_lengths[tail[1:]], budget - produced)
        yield _make_block(tail, sequence_length, pad_token_id, count)


def _make_block(
    tokens: np.ndarray,
    sequence_length: int,
    pad_token_id: int,
    valid_targets: int,
) -> PackedTokenBatch:
    """Turn one short window into padded shifted inputs, targets, and mask."""
    window = np.full(sequence_length + 1, pad_token_id, dtype=np.int64)
    window[: valid_targets + 1] = tokens[: valid_targets + 1]
    window = torch.from_numpy(window)
    target_mask = torch.zeros(sequence_length, dtype=torch.bool)
    target_mask[:valid_targets] = True
    return PackedTokenBatch(window[:-1], window[1:], target_mask)


def iter_packed_blocks(
    split: Dataset,
    sequence_length: int,
    packing_stride: int,
    pad_token_id: int,
    *,
    max_tokens: int | None,
    start_window: int = 0,
    unit_lengths: np.ndarray | None = None,
) -> Iterator[PackedTokenBatch]:
    """Yield fixed-context causal-LM windows from a Hugging Face dataset.

    Documents are concatenated into one causal stream. BOS/EOS tokens mark
    document boundaries, but they do not prevent attention across adjacent
    documents because Fraise's model uses ordinary causal attention. With ``unit_lengths`` the budget counts base-token units of macro targets.
    """
    _check_geometry(sequence_length, packing_stride)
    stream = _stream(split)
    if unit_lengths is not None:
        yield from _iter_unit_blocks(
            stream,
            sequence_length,
            packing_stride,
            pad_token_id,
            max_tokens,
            start_window,
            unit_lengths,
        )
        return
    full_windows = _full_window_count(stream.length, sequence_length, packing_stride)
    budget = max_tokens if max_tokens is not None else float("inf")
    window_span = sequence_length + 1
    # Every window before the last carries a full sequence of targets.
    produced = start_window * sequence_length
    full_mask = torch.ones(sequence_length, dtype=torch.bool)

    window = start_window
    while window < full_windows and produced < budget:
        count = min(full_windows - window, max(1, _GATHER_ELEMENTS // window_span))
        start = window * packing_stride
        tokens = stream.tokens(
            start, start + (count - 1) * packing_stride + window_span
        )
        rows = np.lib.stride_tricks.sliding_window_view(tokens, window_span)
        rows = torch.from_numpy(rows[::packing_stride].astype(np.int64))
        for row in rows:
            valid_targets = min(sequence_length, budget - produced)
            if valid_targets < sequence_length:
                yield _make_block(
                    row.numpy(), sequence_length, pad_token_id, valid_targets
                )
                return
            # Masks of full windows are never mutated, so they share one tensor.
            yield PackedTokenBatch(row[:-1], row[1:], full_mask)
            produced += sequence_length
            if produced >= budget:
                return
        window += count

    if window != full_windows or produced >= budget:
        return
    tail_start = full_windows * packing_stride
    valid_targets = int(
        min(sequence_length, stream.length - tail_start - 1, budget - produced)
    )
    if valid_targets > 0:
        yield _make_block(
            stream.tokens(tail_start, tail_start + valid_targets + 1),
            sequence_length,
            pad_token_id,
            valid_targets,
        )

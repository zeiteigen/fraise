"""Prepare one packed CPU batch while the training thread uses another."""

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

from .packing import PackedTokenBatch


def prefetch_batches(
    batches: Iterable[tuple[int, PackedTokenBatch]],
    *,
    pin_memory: bool,
) -> Iterator[tuple[int, PackedTokenBatch]]:
    """Keep at most one batch ready ahead of the consumer."""
    source = iter(batches)

    def prepare_next() -> tuple[int, PackedTokenBatch]:
        index, batch = next(source)
        if pin_memory:
            batch = PackedTokenBatch(
                input_ids=batch.input_ids.pin_memory(),
                targets=batch.targets.pin_memory(),
                target_mask=batch.target_mask.pin_memory(),
            )
        return index, batch

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(prepare_next)
        while True:
            try:
                item = pending.result()
            except StopIteration:
                return
            pending = executor.submit(prepare_next)
            yield item

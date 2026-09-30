"""Interleave source windows deterministically with bounded memory."""

from random import Random


def iter_mixed_source_blocks(sources, block_counts, *, seed):
    """Shuffle source labels in proportion to their remaining window counts.

    Each source retains its own window order and exact allocation. Only source
    iterators and their remaining counts are stored, rather than all windows.
    """
    active = [
        [name, iter(blocks), block_counts[name]]
        for name, blocks in sources.items()
        if block_counts[name] > 0
    ]
    remaining = sum(count for _, _, count in active)
    random = Random(seed)
    while remaining:
        choice = random.randrange(remaining)
        for index, (name, blocks, count) in enumerate(active):
            if choice < count:
                break
            choice -= count
        try:
            block = next(blocks)
        except StopIteration as exc:
            raise RuntimeError(
                f"dataset {name!r} produced fewer windows than planned"
            ) from exc
        remaining -= 1
        if count == 1:
            active.pop(index)
        else:
            active[index][2] -= 1
        yield name, block

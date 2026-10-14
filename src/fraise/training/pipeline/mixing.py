"""Interleave source windows deterministically with bounded memory."""

from collections.abc import Iterator, Mapping
from random import Random


def mixed_source_order(block_counts: Mapping[str, int], *, seed) -> Iterator[str]:
    """Draw source labels in proportion to their remaining window counts.

    Only labels and remaining counts are stored, so skipping a prefix (to resume)
    costs one random draw per skipped window and no window construction.
    """
    active = [[name, count] for name, count in block_counts.items() if count > 0]
    remaining = sum(count for _, count in active)
    random = Random(seed)
    while remaining:
        choice = random.randrange(remaining)
        for index, (name, count) in enumerate(active):
            if choice < count:
                break
            choice -= count
        remaining -= 1
        if count == 1:
            active.pop(index)
        else:
            active[index][1] -= 1
        yield name


def sequential_source_order(block_counts: Mapping[str, int]) -> Iterator[str]:
    """Every window of each source in turn, in mapping order."""
    for name, count in block_counts.items():
        for _ in range(count):
            yield name


def iter_source_blocks(sources, order: Iterator[str]):
    """Pull the next window from the source each label names."""
    for name in order:
        try:
            block = next(sources[name])
        except StopIteration as exc:
            raise RuntimeError(
                f"dataset {name!r} produced fewer windows than planned"
            ) from exc
        yield name, block


def iter_mixed_source_blocks(sources, block_counts, *, seed):
    """Shuffle source labels while each source retains its own window order."""
    iterators = {name: iter(blocks) for name, blocks in sources.items()}
    return iter_source_blocks(iterators, mixed_source_order(block_counts, seed=seed))

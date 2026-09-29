"""Validate primitive values at configuration boundaries."""

import math


def integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def number(name: str, value: float, *, minimum=0.0, inclusive=False) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)
    ):
        raise ValueError(
            f"{name} must be a finite number {'>=' if inclusive else '>'} {minimum}; got {value!r}"
        )

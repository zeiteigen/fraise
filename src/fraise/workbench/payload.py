"""Fit workbench results through AblationKit's view bridge, which carries at most 256 KiB.

Results travel as ``{"encoding": "gzip+base64:packed-json", "payload": ...}``: JSON in which a
list of objects with the same keys becomes ``{"__cols__", "__rows__"}``, gzipped and base64
encoded. Nothing is dropped; the views' ``codec.ts`` undoes it.
"""

import base64
import gzip
import json
from typing import Any

ENCODING = "gzip+base64:packed-json"
MAX_BYTES = 250_000


def pack(value: Any) -> Any:
    """Lossless: a list of objects with the same keys becomes {"__cols__", "__rows__"}."""
    if isinstance(value, list):
        items = [pack(item) for item in value]
        if (
            len(items) > 1
            and all(isinstance(item, dict) and "__cols__" not in item for item in items)
            and len({tuple(item) for item in items}) == 1
        ):
            return {"__cols__": list(items[0]), "__rows__": [list(item.values()) for item in items]}
        return items
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    return value


def encode(value: Any, *, too_large: str = "shorten the request") -> dict[str, str]:
    raw = json.dumps(pack(value), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    if len(payload) > MAX_BYTES:
        raise ValueError(
            f"this result is {len(payload):,} bytes compressed, over the {MAX_BYTES:,}-byte "
            f"limit for results shown in a view: {too_large}"
        )
    return {"encoding": ENCODING, "payload": payload}

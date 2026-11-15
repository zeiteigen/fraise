"""fraise-web's Autocomplete, Compare and Train views, run through fraise's own code.

Everything is `fraise.workbench`: the catalog of saved runs (`Catalog`), generation
(`generate_sequentially` with an `InferenceSession` that keeps the last checkpoint loaded, as
fraise-web's server did), training validation and launch plans (`validate_training`,
`write_plan`) and saved inputs. This module only adapts AblationKit params and results.

The worker is warm, so the loaded model and the tokenizer fingerprints stay between runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import monotonic
from typing import Any

from ablationkit_extension import ExtensionContext, ExtensionResult

from fraise.paths import data_root
from fraise.workbench.catalog import Catalog
from fraise.workbench.generation import InferenceSession, generate_sequentially
from fraise.workbench.payload import encode
from fraise.workbench.requests import GenerationRequest, TrainingRequest
from fraise.workbench.saved_inputs import SavedInputs
from fraise.workbench.training import (
    resolve_device,
    training_catalog,
    validate_training,
    write_plan,
)

PROGRESS_CHARS = 480  # AblationKit keeps 500 characters of a progress message
PROGRESS_SECONDS = 0.2

_STATE: dict[str, Any] = {}


def _checkout() -> Path:
    return Path(os.environ.get("ABLATIONKIT_CHECKOUT", ".")).resolve()


def _catalog() -> Catalog:
    if "catalog" not in _STATE:
        _STATE["catalog"] = Catalog(data_root(_checkout() / "data"))
    return _STATE["catalog"]


def _device() -> str:
    if "device" not in _STATE:
        _STATE["device"] = resolve_device("auto")
    return _STATE["device"]


def _session() -> InferenceSession:
    if "session" not in _STATE:
        _STATE["session"] = InferenceSession(_device())
    return _STATE["session"]


def _device_info() -> dict[str, Any]:
    import torch

    return {"resolved": _device(), "cuda_available": torch.cuda.is_available()}


def runs(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """Saved runs with their checkpoints, configuration, tokenizer and validation (GET /api/runs)."""
    catalog = _catalog()
    records = catalog.runs()
    data = {"data_dir": str(catalog.data_dir), "models_dir": str(catalog.models_dir)}
    data |= {"device": _device_info(), "runs": records}
    usable = sum(1 for run in records if run.get("checkpoints"))
    return ExtensionResult(
        summary=f"{len(records)} runs, {usable} with checkpoints, in {catalog.models_dir}",
        data=encode(data, too_large="too many saved runs to list"),
    )


def _progress_message(payload: dict[str, Any]) -> str:
    """At most PROGRESS_CHARS characters: the output's length and as much of its end as fits."""
    text = payload.get("text", "")
    tail = text[-PROGRESS_CHARS:]
    while True:
        message = json.dumps(
            {
                "i": payload["index"],
                "n": payload.get("generated_tokens", 0),
                "len": len(text),
                "tail": tail,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(message) <= 500 or not tail:
            return message
        tail = tail[len(tail) // 4 or 1 :]


def generate(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """POST /api/generate: each checkpoint continues the raw prompt with the same seed, in turn.

    While a checkpoint generates, progress messages carry its output's length and end, so the
    view can stream it; the result holds every output in full.
    """
    request = GenerationRequest.parse(params)
    catalog = _catalog()
    references = [(ref, *catalog.resolve(ref)) for ref in request.models]
    count, last = len(references), [0.0]
    error: dict[str, Any] = {}

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        index = payload.get("index") or 0
        if kind == "loading":
            ctx.progress(
                int(100 * index / count),
                json.dumps({"i": index, "phase": "loading"}, separators=(",", ":")),
            )
        elif kind == "output" and monotonic() - last[0] >= PROGRESS_SECONDS:
            last[0] = monotonic()
            done = payload["generated_tokens"] / max(request.max_new_tokens, 1)
            ctx.progress(int(100 * (index + min(done, 1)) / count), _progress_message(payload))
        elif kind == "error":
            error.update(payload)

    results = generate_sequentially(request, references, _session(), on_event)
    data = {"results": results, "error": error or None}
    summary = "; ".join(
        f"{r['run_id']}/{r['checkpoint_id']}: {r['generated_tokens']} tokens, "
        f"{r['tokens_per_second']:.1f} tok/s"
        for r in results
    )
    if error:
        summary = f"{summary}; failed at #{error.get('index')}: {error.get('message')}".lstrip("; ")
    return ExtensionResult(
        summary=summary or "nothing generated",
        metrics={f"tokens_per_second/{i}": r["tokens_per_second"] for i, r in enumerate(results)},
        data=encode(data, too_large="generate fewer tokens or compare fewer checkpoints"),
    )


def release(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """Unload the loaded checkpoint and return its device memory (fraise-web did this before training)."""
    session = _session()
    held = session.key[0] if session.key else None
    session.unload()
    return ExtensionResult(summary=f"released {held}" if held else "no checkpoint was loaded")


def train_catalog(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """Architectures and their fields, tokenizers, sources and defaults (GET /api/catalog)."""
    data = training_catalog(_catalog(), _device_info())
    return ExtensionResult(
        summary=f"{len(data['architectures'])} architectures, {len(data['tokenizers'])} tokenizers",
        data=encode(data, too_large="too many tokenizers to list"),
    )


def validate(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """POST /api/config/validate: the normalized configuration and its parameter count."""
    normalized, _ = validate_training(TrainingRequest.parse(params), _catalog(), _device())
    return ExtensionResult(
        summary=f"valid: {normalized['parameter_count']:,} parameters",
        metrics={"parameter_count": normalized["parameter_count"]},
        data=encode(normalized),
    )


def plan(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """Check a run and write its launch plan for the MLproject `launch` entrypoint."""
    catalog = _catalog()
    written = write_plan(
        TrainingRequest.parse(params), catalog, _device(), catalog.data_dir / "launch"
    )
    directory = Path(written["directory"])
    checkout = _checkout()
    written["parameter"] = (
        str(directory.relative_to(checkout)) if directory.is_relative_to(checkout) else str(directory)
    )
    return ExtensionResult(
        summary=f"{written['mode']} plan for {written['run_name']}: {written['parameter']}",
        metrics={"parameter_count": written["parameter_count"]},
        data=written,
    )


def inputs(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """Saved Autocomplete inputs: `list` (summaries), `get` (one in full), `save` and `clear`."""
    store = SavedInputs(_catalog().data_dir / "workspace" / "saved-inputs.json")
    action = params.get("action", "list")
    if action == "get":
        item = next((i for i in store.items() if i["id"] == params.get("id")), None)
        if item is None:
            raise ValueError("that saved input no longer exists")
        return ExtensionResult(summary=item["id"], data=encode(item))
    if action == "save":
        item = store.remember(params.get("item"))
    elif action == "clear":
        store.clear()
    elif action != "list":
        raise ValueError(f"unknown action {action!r}")
    summaries = [
        {
            "id": i["id"],
            "title": i["text"][:60],
            "run_id": i["ref"].get("run_id"),
            "checkpoint_id": i["ref"].get("checkpoint_id"),
            "updated": i.get("updated"),
        }
        for i in store.items()
    ]
    data = {"items": summaries}
    if action == "save":
        data["saved"] = item["id"]
    return ExtensionResult(summary=f"{len(summaries)} saved inputs", data=data)

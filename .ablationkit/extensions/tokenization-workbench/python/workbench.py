"""The tokenization workbench, run through fraise's own code in the project environment.

Nothing is reimplemented here: the catalog is `fraise.workbench.catalog.Catalog`, requests are
checked by `fraise.workbench.requests.TokenizeRequest` and analysis is
`fraise.workbench.tokenization.TokenizationWorkbench.analyze` (which calls
`fraise.preview.hierarchy.analyze_text` and the hierarchy constructors). This module only adapts
AblationKit params and results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ablationkit_extension import ExtensionContext, ExtensionResult

from fraise.paths import data_root
from fraise.workbench.catalog import Catalog
from fraise.workbench.payload import encode
from fraise.workbench.requests import TokenizeRequest
from fraise.workbench.tokenization import TokenizationWorkbench

# The PoC pilot's three variants, each with the hierarchy its own run was trained with.
DEFAULT_CONFIGURATIONS = [
    {"label": "8k/8k BPE (pilot-bpe8k)", "hierarchy": {"kind": "run", "run": {"run_id": "pilot-bpe8k"}}},
    {"label": "64k/64k macro (pilot-macro64k)", "hierarchy": {"kind": "run", "run": {"run_id": "pilot-macro64k"}}},
    {"label": "Fraise macro-64k in / 8k out (pilot-fraise)", "hierarchy": {"kind": "run", "run": {"run_id": "pilot-fraise"}}},
]


def _catalog() -> Catalog:
    checkout = Path(os.environ.get("ABLATIONKIT_CHECKOUT", "."))
    return Catalog(data_root(checkout / "data"))


_WORKBENCH: TokenizationWorkbench | None = None


def _workbench() -> TokenizationWorkbench:
    """One workbench per (warm) worker: its tokenizer and hierarchy caches survive between
    analyses, keyed by file identity so edits are picked up."""
    global _WORKBENCH
    if _WORKBENCH is None:
        _WORKBENCH = TokenizationWorkbench(_catalog())
    return _WORKBENCH


_CATALOG: tuple[tuple[Any, ...], dict[str, Any]] | None = None


def _catalog_key(cat: Catalog) -> tuple[Any, ...]:
    """What the catalog reads: run directories (their entries, e.g. checkpoints), their
    metadata, the tokenizer files and the hierarchy directory. A change to any of them changes
    the key, so a cached catalog is never stale."""

    def stat(path: Path) -> tuple[str, int, int]:
        try:
            s = path.stat()
            return (str(path), s.st_mtime_ns, s.st_size)
        except OSError:
            return (str(path), 0, 0)

    parts: list[tuple[str, int, int]] = [stat(cat.models_dir), stat(cat.data_dir / "tokenizers")]
    parts.append(stat(cat.data_dir / "hierarchies"))
    for meta in sorted(cat.models_dir.glob("*/model_meta.json")):
        parts += [stat(meta.parent), stat(meta)]
    for tok in sorted((cat.data_dir / "tokenizers").glob("*.json")):
        parts.append(stat(tok))
    return tuple(parts)


def catalog(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """Tokenizers, saved runs and hierarchy files, for the view's selectors (like GET /api/catalog).
    Describing every run fingerprints its tokenizer (seconds), so the warm worker keeps the
    answer until a file it was built from changes."""
    global _CATALOG
    cat = _workbench().catalog
    key = _catalog_key(cat)
    if _CATALOG is not None and _CATALOG[0] == key:
        data = _CATALOG[1]
        return ExtensionResult(summary=_catalog_summary(data), data=data)
    runs = [
        {
            "id": run["id"],
            "name": run["name"],
            "architecture": run.get("architecture"),
            "checkpoints": [c["id"] for c in run.get("checkpoints", [])],
            "default_checkpoint": run.get("default_checkpoint"),
            "error": run.get("error"),
        }
        for run in cat.runs()
    ]
    tokenizers = [
        {k: t[k] for k in ("id", "label", "vocab_size", "macro", "base_vocab_size")}
        for t in cat.tokenizers()
    ]
    hierarchies = _workbench().hierarchy_files()
    data = {
        "data_dir": str(cat.data_dir),
        "tokenizers": tokenizers,
        "runs": runs,
        "hierarchies": hierarchies,
        "defaults": DEFAULT_CONFIGURATIONS,
    }
    _CATALOG = (key, data)
    return ExtensionResult(summary=_catalog_summary(data), data=data)


def _catalog_summary(data: dict[str, Any]) -> str:
    return (
        f"{len(data['tokenizers'])} tokenizers, {len(data['runs'])} runs, "
        f"{len(data['hierarchies'])} hierarchy files"
    )


def tokenize(ctx: ExtensionContext, params: dict[str, Any]) -> ExtensionResult:
    """One analysis per configuration (fraise-web's POST /api/tokenize)."""
    request = TokenizeRequest.parse(
        {**params, "configurations": params.get("configurations") or DEFAULT_CONFIGURATIONS}
    )
    ctx.progress(10, "loading tokenizers and hierarchies")
    result = _workbench().analyze(request)
    analyses = result["results"]
    return ExtensionResult(
        summary="; ".join(
            f"{a.get('name')}: "
            + (a["error"] if "error" in a else f"{len(a['input_tokens'])} positions")
            for a in analyses
        ),
        metrics={
            f"positions/{i}": len(a["input_tokens"])
            for i, a in enumerate(analyses)
            if "error" not in a
        },
        # The bridge carries at most 256 KiB; the analysis arrives packed and gzipped, losslessly.
        data=encode(
            result,
            too_large=f"shorten the text ({len(request.text):,} characters now) "
            "or compare fewer configurations",
        ),
    )

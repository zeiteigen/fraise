"""Checked workbench requests, parsed from plain JSON-like data.

These keep the bounds fraise-web's API enforced: unknown fields are refused, integers must be
integers (not booleans), and every limit matches the old request schemas.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

HIERARCHY_KINDS = ("auto", "none", "merges", "file", "run")


def _mapping(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name}: unknown field {unknown[0]!r}")
    return value


def _string(value: Any, name: str, *, min_length=0, max_length=None, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if len(value) < min_length or (max_length is not None and len(value) > max_length):
        bounds = f"{min_length}-{max_length}" if max_length else f"at least {min_length}"
        raise ValueError(f"{name} must be {bounds} characters")
    return value


def _integer(value: Any, name: str, *, low=None, high=None, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _number(value: Any, name: str, *, low, high, low_open=False, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < low or (low_open and value == low) or value > high:
        raise ValueError(f"{name} must be {'above' if low_open else 'at least'} {low} and at most {high}")
    return float(value)


def _integers(value: Any, name: str):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError(f"{name} must be a list of at most 16 integers")
    return [_integer(item, name) for item in value]


@dataclass(frozen=True)
class ModelReference:
    run_id: str
    checkpoint_id: str | None = None

    @classmethod
    def parse(cls, data: Any, name: str = "model") -> "ModelReference":
        data = _mapping(data, name, {"run_id", "checkpoint_id"})
        return cls(
            _string(data.get("run_id"), f"{name}.run_id", min_length=1),
            _string(data.get("checkpoint_id"), f"{name}.checkpoint_id", optional=True),
        )


@dataclass(frozen=True)
class GenerationRequest:
    """Raw text continuation: no roles, messages or chat templates."""

    models: tuple[ModelReference, ...]
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = 0.95
    seed: int = 1337

    @classmethod
    def parse(cls, data: Any) -> "GenerationRequest":
        data = _mapping(
            data,
            "request",
            {"models", "prompt", "max_new_tokens", "temperature", "top_k", "top_p", "seed"},
        )
        models = data.get("models")
        if not isinstance(models, list) or not 1 <= len(models) <= 8:
            raise ValueError("choose 1-8 checkpoints")
        return cls(
            models=tuple(
                ModelReference.parse(item, f"models[{index}]") for index, item in enumerate(models)
            ),
            prompt=_string(data.get("prompt"), "prompt", min_length=1, max_length=100000),
            max_new_tokens=_integer(
                data.get("max_new_tokens", 128), "max_new_tokens", low=0, high=4096
            ),
            temperature=_number(data.get("temperature", 0.0), "temperature", low=0, high=100),
            top_k=_integer(data.get("top_k"), "top_k", low=1, optional=True),
            top_p=_number(
                data.get("top_p", 0.95), "top_p", low=0, high=1, low_open=True, optional=True
            ),
            seed=_integer(data.get("seed", 1337), "seed", low=0, high=2**63 - 1),
        )


@dataclass(frozen=True)
class HierarchySpec:
    """``auto`` shows a macro tokenizer's two-level hierarchy (8k -> 64k)."""

    kind: str = "auto"
    level_sizes: list[int] | None = None
    residual_dims: list[int] | None = None
    file_id: str | None = None
    run: ModelReference | None = None

    @classmethod
    def parse(cls, data: Any, name: str = "hierarchy") -> "HierarchySpec":
        data = _mapping(
            data or {}, name, {"kind", "level_sizes", "residual_dims", "file_id", "run"}
        )
        kind = data.get("kind", "auto")
        if kind not in HIERARCHY_KINDS:
            raise ValueError(f"unknown hierarchy kind {kind!r}")
        run = data.get("run")
        return cls(
            kind=kind,
            level_sizes=_integers(data.get("level_sizes"), "level_sizes"),
            residual_dims=_integers(data.get("residual_dims"), "residual_dims"),
            file_id=_string(data.get("file_id"), f"{name}.file_id", optional=True),
            run=ModelReference.parse(run, f"{name}.run") if run is not None else None,
        )


@dataclass(frozen=True)
class TokenizeConfiguration:
    label: str | None = None
    tokenizer_id: str | None = None
    hierarchy: HierarchySpec = field(default_factory=HierarchySpec)

    @classmethod
    def parse(cls, data: Any, name: str = "configuration") -> "TokenizeConfiguration":
        data = _mapping(data, name, {"label", "tokenizer_id", "hierarchy"})
        return cls(
            label=_string(data.get("label"), f"{name}.label", max_length=200, optional=True),
            tokenizer_id=_string(data.get("tokenizer_id"), f"{name}.tokenizer_id", optional=True),
            hierarchy=HierarchySpec.parse(data.get("hierarchy"), f"{name}.hierarchy"),
        )


@dataclass(frozen=True)
class TokenizeRequest:
    text: str
    configurations: tuple[TokenizeConfiguration, ...]
    add_special_tokens: bool = False

    @classmethod
    def parse(cls, data: Any) -> "TokenizeRequest":
        data = _mapping(data, "request", {"text", "add_special_tokens", "configurations"})
        configurations = data.get("configurations")
        if not isinstance(configurations, list) or not 1 <= len(configurations) <= 6:
            raise ValueError("give 1-6 configurations")
        special = data.get("add_special_tokens", False)
        if not isinstance(special, bool):
            raise ValueError("add_special_tokens must be true or false")
        return cls(
            text=_string(data.get("text"), "text", min_length=1, max_length=20000),
            configurations=tuple(
                TokenizeConfiguration.parse(item, f"configurations[{index}]")
                for index, item in enumerate(configurations)
            ),
            add_special_tokens=special,
        )


@dataclass(frozen=True)
class TrainingRequest:
    """A fresh run (``resume`` unset) or a resumed one, as the Train panel submits it."""

    model_config: dict
    training_config: dict
    tokenizer_id: str
    run_name: str
    resume: ModelReference | None = None

    @classmethod
    def parse(cls, data: Any) -> "TrainingRequest":
        data = _mapping(
            data,
            "request",
            {"model_config", "training_config", "tokenizer_id", "run_name", "resume"},
        )
        for name in ("model_config", "training_config"):
            if not isinstance(data.get(name), dict):
                raise ValueError(f"{name} must be an object")
        resume = data.get("resume")
        return cls(
            model_config=data["model_config"],
            training_config=data["training_config"],
            tokenizer_id=_string(data.get("tokenizer_id"), "tokenizer_id", min_length=1),
            run_name=_string(data.get("run_name"), "run_name", min_length=1, max_length=100),
            resume=ModelReference.parse(resume, "resume") if resume is not None else None,
        )

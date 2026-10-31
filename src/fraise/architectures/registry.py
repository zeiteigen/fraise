"""Resolve architecture IDs without importing the baseline during config import."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

from .contracts import ArchitectureSpec, ModelConfigProtocol, ModelProtocol

_architectures: dict[str, ArchitectureSpec] = {}


def _load_builtin() -> None:
    if "transformer" not in _architectures:
        from .hierarchical import FRAISE
        from .transformer import TRANSFORMER

        _architectures[TRANSFORMER.id] = TRANSFORMER
        _architectures[FRAISE.id] = FRAISE


def register_architecture(spec: ArchitectureSpec) -> None:
    """Register a frozen config dataclass and its hooks under a unique ID.

    Config types and hooks must be picklable for spawned checkpoint validation;
    define them at module scope rather than inside a registration function.
    """
    _load_builtin()
    if not isinstance(spec.id, str) or not spec.id:
        raise ValueError("architecture ID must be a non-empty string")
    if spec.id in _architectures:
        raise ValueError(f"architecture {spec.id!r} is already registered")
    config_type = spec.config_type
    if (
        not isinstance(config_type, type)
        or not is_dataclass(config_type)
        or not config_type.__dataclass_params__.frozen
    ):
        raise TypeError("architecture config_type must be a frozen dataclass")
    required = {"architecture", "vocab_size", "max_seq_len"}
    if required - {field.name for field in fields(config_type)}:
        raise TypeError(
            "architecture config_type requires architecture, vocab_size, max_seq_len"
        )
    _architectures[spec.id] = spec


def get_architecture(architecture: str) -> ArchitectureSpec:
    _load_builtin()
    if not isinstance(architecture, str):
        raise TypeError("architecture must be a string")
    try:
        return _architectures[architecture]
    except KeyError:
        raise ValueError(f"unknown architecture {architecture!r}") from None


def list_architectures() -> tuple[ArchitectureSpec, ...]:
    _load_builtin()
    return tuple(_architectures.values())


def model_config_from_dict(values: Mapping[str, object]) -> ModelConfigProtocol:
    """Parse architecture-specific settings; omitted IDs select the baseline."""
    if not isinstance(values, Mapping):
        raise TypeError("model configuration must be a JSON object")
    values = dict(values)
    values.setdefault("architecture", "transformer")
    spec = get_architecture(values["architecture"])
    return spec.config_type(**values)


def build_model(config: ModelConfigProtocol) -> ModelProtocol:
    spec = get_architecture(config.architecture)
    if not isinstance(config, spec.config_type):
        raise TypeError(
            f"architecture {spec.id!r} requires {spec.config_type.__name__}"
        )
    return spec.build(config)

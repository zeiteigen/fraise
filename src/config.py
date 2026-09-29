"""Dataset and JSON configuration loading for Fraise training scripts."""

from collections.abc import Callable, Collection, Mapping
import json
import math
from pathlib import Path
from tokenizers import Tokenizer
from typing import Any, TypeAlias, cast

from components.model import ModelConfig
from datasets_mgr import unified_dataset_loader
from datasets_mgr.babylm import DATASET_NAME as BABYLM_NAME, use_babylm
from datasets_mgr.finewebedu import DATASET_NAME as FINEWEB_NAME, use_fineweb_edu
from datasets_mgr.cosmopedia_khan import (
    DATASET_NAME as KHAN_NAME,
    use_cosmopedia_khan_academy,
)
from datasets_mgr.cosmopedia_stories import (
    DATASET_NAME as STORIES_NAME,
    use_cosmopedia_stories,
)


JsonConfig: TypeAlias = dict[str, Any]
ConfigPath: TypeAlias = str | Path
JsonTypeCheck: TypeAlias = Callable[[object], bool]


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_boolean(value: object) -> bool:
    return isinstance(value, bool)


def _is_optional_integer(value: object) -> bool:
    return value is None or _is_integer(value)


def _is_betas(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_number(beta) for beta in value)
    )


def _is_dataset_token_counts(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(dataset_name, str)
            and bool(dataset_name)
            and _is_integer(count)
            for dataset_name, count in value.items()
        )
    )


_TRAINING_HYPERPARAMETER_TYPES: dict[str, JsonTypeCheck] = {
    "num_epochs": _is_integer,
    "batch_size": _is_integer,
    "learning_rate": _is_number,
    "betas": _is_betas,
    "weight_decay": _is_number,
    "dataset_token_counts": _is_dataset_token_counts,
    "warmup_steps": _is_integer,
    "sequence_length": _is_integer,
    "max_train_tokens": _is_integer,
    "max_validation_tokens": _is_optional_integer,
    "packing_stride": _is_integer,
    "seed": _is_integer,
    "checkpoint_interval_steps": _is_integer,
    "tokenization_batch_size": _is_integer,
    "use_bf16": _is_boolean,
    "shuffle": _is_boolean,
    "gradient_clip_norm": _is_number,
}

_OPTIONAL_TRAINING_HYPERPARAMETER_TYPES: dict[str, JsonTypeCheck] = {
    "minibatch_size": _is_integer,
    "gradient_accumulation_steps": _is_integer,
    "checkpoint_keep_last": _is_integer,
}

_MODEL_CONFIG_TYPES: dict[str, JsonTypeCheck] = {
    "vocab_size": _is_integer,
    "residual_stream_dim": _is_integer,
    "head_dim": _is_integer,
    "qk_dim": _is_integer,
    "num_heads": _is_integer,
    "swiglu_up_dim": _is_integer,
    "num_layers": _is_integer,
    "max_seq_len": _is_integer,
}

_OPTIONAL_MODEL_CONFIG_TYPES: dict[str, JsonTypeCheck] = {
    "tie_word_embeddings": _is_boolean,
}


def _load_json_object(path: ConfigPath) -> JsonConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {config_path}: {error.msg}") from error

    if not isinstance(config, dict):
        raise TypeError(
            f"configuration file must contain a JSON object: {config_path}"
        )
    return config


def _validate_config_types(
    config: JsonConfig,
    expected_types: dict[str, JsonTypeCheck],
    config_name: str,
    optional_types: dict[str, JsonTypeCheck] | None = None,
) -> JsonConfig:
    optional_types = optional_types or {}
    expected_keys = set(expected_types)
    actual_keys = set(config)
    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        raise TypeError(
            f"{config_name} is missing required keys: {', '.join(missing_keys)}"
        )

    unexpected_keys = sorted(actual_keys - expected_keys - set(optional_types))
    if unexpected_keys:
        raise TypeError(
            f"{config_name} contains unexpected keys: {', '.join(unexpected_keys)}"
        )

    for key, type_check in (expected_types | optional_types).items():
        if key not in config:
            continue
        value = config[key]
        if not type_check(value):
            raise TypeError(
                f"{config_name}['{key}'] has an invalid type: "
                f"expected a JSON-compatible value of the configured type, "
                f"got {type(value).__name__}"
            )
    return config


def load_json_config(path: ConfigPath) -> JsonConfig:
    """Load a JSON object without applying a configuration-specific schema."""
    return _load_json_object(path)


def load_training_hyperparameters(path: ConfigPath) -> JsonConfig:
    """Load and type-check the training hyperparameters JSON file."""
    config = _load_json_object(path)
    return _validate_config_types(
        config,
        _TRAINING_HYPERPARAMETER_TYPES,
        "training hyperparameters",
        _OPTIONAL_TRAINING_HYPERPARAMETER_TYPES,
    )


def load_model_config(path: ConfigPath) -> ModelConfig:
    """Load and validate model dimensions and options from a JSON file."""
    config = _load_json_object(path)
    _validate_config_types(
        config, _MODEL_CONFIG_TYPES, "model config", _OPTIONAL_MODEL_CONFIG_TYPES
    )
    for name in _MODEL_CONFIG_TYPES:
        if config[name] <= 0:
            raise ValueError(f"model config['{name}'] must be positive")
    if config["qk_dim"] % 2:
        raise ValueError("model config['qk_dim'] must be even for RoPE")
    return cast(ModelConfig, config)


DATASET_LOADERS = {
    BABYLM_NAME: use_babylm,
    FINEWEB_NAME: use_fineweb_edu,
    KHAN_NAME: use_cosmopedia_khan_academy,
    STORIES_NAME: use_cosmopedia_stories,
}


def get_dataset_loader(
    dataset_names: Collection[str] | None = None,
    *,
    tokenizer: Tokenizer | None = None,
    dataset_token_counts: Mapping[str, int] | None = None,
):
    """Load the requested training sources and retain FineWeb validation."""
    selected = set(DATASET_LOADERS) if dataset_names is None else set(dataset_names)
    unknown = selected - DATASET_LOADERS.keys()
    if unknown:
        raise ValueError(f"unknown dataset name(s): {', '.join(sorted(unknown))}")
    if (tokenizer is None) != (dataset_token_counts is None):
        raise ValueError("tokenizer and dataset_token_counts must be supplied together")
    if dataset_token_counts is not None:
        missing = selected - dataset_token_counts.keys()
        if missing:
            raise ValueError(
                "dataset_token_counts is missing selected source(s): "
                + ", ".join(sorted(missing))
            )

    sources = []
    for name, loader in DATASET_LOADERS.items():
        if name in selected:
            if dataset_token_counts is not None and name != BABYLM_NAME:
                sources.append(
                    loader(
                        max_train_tokens=dataset_token_counts[name],
                        tokenizer=tokenizer,
                    )
                )
            else:
                sources.append(loader())
        elif name == FINEWEB_NAME and dataset_names is not None:
            sources.append(loader(include_train=False))
    return unified_dataset_loader(
        sources,
        source_column=dataset_token_counts is None,
    )

"""Read configuration JSON into the authoritative normalized definitions."""

from fraise.io import read_json

from .model import ModelConfig
from .training import TrainingSettings


def load_model_config(path) -> ModelConfig:
    return ModelConfig(**_read_values(path))


def load_training_settings(path) -> TrainingSettings:
    return TrainingSettings.from_dict(_read_values(path))


def _read_values(path):
    values = read_json(path)
    if not isinstance(values, dict):
        raise TypeError("configuration must contain a JSON object")
    return values

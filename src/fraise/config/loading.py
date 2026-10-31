"""Read configuration JSON into the authoritative normalized definitions."""

from fraise.architectures import ModelConfigProtocol, model_config_from_dict
from fraise.io import read_json

from .training import TrainingSettings


def load_model_config(path) -> ModelConfigProtocol:
    return model_config_from_dict(_read_values(path))


def load_training_settings(path) -> TrainingSettings:
    return TrainingSettings.from_dict(_read_values(path))


def _read_values(path):
    values = read_json(path)
    if not isinstance(values, dict):
        raise TypeError("configuration must contain a JSON object")
    return values

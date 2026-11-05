"""Present architecture settings and validate submitted training configurations."""

from copy import deepcopy
from dataclasses import asdict, fields

import torch

from fraise.architectures import list_architectures, model_config_from_dict
from fraise.config.compatibility import check_compatibility
from fraise.config.training import TrainingSettings
from fraise.data.sources import SOURCE_NAMES
from fraise.io import read_json
from fraise.model.parameters import estimate_parameter_count
from fraise.tokenizer.loading import load_tokenizer, tokenizer_fingerprint
from fraise.training.run.manifest import check_resume_identity

from .catalog import artifact_name, contained


def training_defaults():
    return asdict(
        TrainingSettings.from_dict(
            {
                "batch_size": 8,
                "minibatch_size": 2,
                "learning_rate": 0.02,
                "num_epochs": 1,
                "max_train_tokens": 32768,
                "dataset_token_counts": {
                    name: 32768 if name == "fineweb-edu" else 0 for name in SOURCE_NAMES
                },
                "sequence_length": 128,
                "use_compile": False,
                "max_validation_tokens": 0,
                "log_interval_steps": 10,
            }
        )
    )


def describe_architectures():
    records = []
    for spec in list_architectures():
        example = dict(spec.example_config)
        descriptions = []
        for field in fields(spec.config_type):
            value = example.get(field.name)
            if isinstance(value, bool):
                kind = "boolean"
            elif isinstance(value, (int, float)):
                kind = "number"
            elif isinstance(value, dict):
                kind = "object"
            elif isinstance(value, (list, tuple)):
                kind = "array"
            elif value is None and field.default is None:
                # Optional nested settings (such as embedding_hierarchy) are JSON-only.
                kind = "object"
            else:
                kind = "string"
            descriptions.append({"name": field.name, "type": kind, "default": value})
        records.append(
            {
                "id": spec.id,
                "label": spec.label,
                "example": example,
                "fields": descriptions,
            }
        )
    return records


def validate_training(request, catalog, device):
    config = model_config_from_dict(request.configuration)
    settings = TrainingSettings.from_dict(request.training_config)
    settings.validate_sources(SOURCE_NAMES)
    tokenizer = load_tokenizer(catalog.tokenizer_path(request.tokenizer_id))
    check_compatibility(tokenizer, config, settings)
    if settings.use_bf16 and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("use_bf16 requires a CUDA device with BF16 support")
    artifact_name(request.run_name)
    resume_path = None
    if request.resume is not None:
        directory, resume_path = catalog.resolve(request.resume, resume=True)
        metadata = read_json(directory / "model_meta.json")
        if request.run_name != metadata["model_name"]:
            raise ValueError("resume run name must match the saved run")
        saved = metadata["manifest"]
        current = deepcopy(saved)
        current["identity"].update(
            {
                "model_config": asdict(config),
                "training": asdict(settings),
                "tokenizer": {
                    "fingerprint": tokenizer_fingerprint(tokenizer),
                    "vocab_size": tokenizer.get_vocab_size(),
                },
            }
        )
        check_resume_identity(saved, current)
    else:
        destination = contained(
            catalog.models_dir / request.run_name, catalog.models_dir
        )
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("run directory is not empty; choose a new name or resume")
    return {
        "valid": True,
        "model_config": asdict(config),
        "training_config": asdict(settings),
        "parameter_count": estimate_parameter_count(config),
    }, resume_path

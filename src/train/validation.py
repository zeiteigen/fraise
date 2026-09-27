"""Token-weighted validation, including the separate CPU worker."""

from dataclasses import replace

import torch
from datasets import Dataset

from components.model import FraiseModel, ModelConfig

from .batch import _autocast_context, _loss_for_batch, _move_batch
from .hyperparameters import TrainingSettings
from .packing import iter_packed_batches


def _evaluate(
    model: FraiseModel,
    split: Dataset,
    settings: TrainingSettings,
    pad_token_id: int,
) -> float | None:
    """Compute token-weighted loss on the validation split."""
    total_loss = 0.0
    total_tokens = 0
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for _, cpu_batch in iter_packed_batches(
            split,
            settings.sequence_length,
            settings.packing_stride,
            pad_token_id,
            batch_size=settings.batch_size,
            seed=settings.seed,
            shuffle=False,
            max_tokens=settings.max_validation_tokens,
        ):
            batch = _move_batch(cpu_batch, model_device)
            with _autocast_context(settings.use_bf16):
                result = _loss_for_batch(model, batch)
            if result is None:
                continue

            loss, valid_tokens = result
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens

    return total_loss / total_tokens if total_tokens else None


_validation_model: FraiseModel | None = None
_validation_split: Dataset | None = None
_validation_settings: TrainingSettings | None = None
_validation_pad_token_id: int | None = None


def _init_validation_worker(
    model_config: ModelConfig,
    split: Dataset,
    settings: TrainingSettings,
    pad_token_id: int,
) -> None:
    """Build a separate CPU model once in the validation process."""
    global _validation_model, _validation_split, _validation_settings
    global _validation_pad_token_id
    torch.set_num_threads(1)
    _validation_model = FraiseModel(model_config).to("cpu")
    _validation_model.eval()
    _validation_split = split
    _validation_settings = replace(settings, use_bf16=False)
    _validation_pad_token_id = pad_token_id


def _evaluate_checkpoint(checkpoint_path: str) -> float | None:
    """Read a completed checkpoint and evaluate it without using the GPU."""
    if (
        _validation_model is None
        or _validation_split is None
        or _validation_settings is None
        or _validation_pad_token_id is None
    ):
        raise RuntimeError("validation worker was not initialized")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validation_model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    return _evaluate(
        _validation_model,
        _validation_split,
        _validation_settings,
        _validation_pad_token_id,
    )

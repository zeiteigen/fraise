"""Construct the training runtime and execute one optimizer step."""

import math
from dataclasses import dataclass

import torch
from transformers.optimization import get_cosine_schedule_with_warmup

from fraise.model.model import FraiseModel

from .minibatches import backward_minibatches
from .rng import seed_everything


@dataclass
class TrainingRuntime:
    model: FraiseModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: torch.device

    def step(self, batch, settings):
        self.optimizer.zero_grad(set_to_none=True)
        loss, tokens = backward_minibatches(
            self.model,
            batch,
            self.device,
            minibatch_size=settings.minibatch_size,
            use_bf16=settings.use_bf16,
        )
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), settings.gradient_clip_norm
        )
        loss, norm = torch.stack((loss.detach(), norm.detach())).tolist()
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise FloatingPointError("non-finite training loss or gradient norm")
        self.optimizer.step()
        self.scheduler.step()
        return loss, tokens, norm


def build_runtime(
    config,
    settings,
    steps,
    device: torch.device | None = None,
    *,
    initialize_weights=True,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if settings.use_bf16:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("use_bf16 requires a CUDA device with BF16 support")
        torch.set_float32_matmul_precision("high")
    seed_everything(settings.seed)
    model = FraiseModel(config)
    if initialize_weights and config.tie_word_embeddings:
        # Unit-scale embeddings make the residual stream predict its own token.
        torch.nn.init.normal_(
            model.token_embedding.weight, std=settings.tied_weight_init_std
        )
    model.to(device)
    model.compile()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=settings.betas,
        weight_decay=settings.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, settings.warmup_steps, steps)
    return TrainingRuntime(model, optimizer, scheduler, device)

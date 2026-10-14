"""Construct the training runtime and execute one optimizer step."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers.optimization import get_cosine_schedule_with_warmup

from fraise.config.training import TrainingSettings
from fraise.model.model import FraiseModel

from .batch import compile_loss_sum, token_loss_sum
from .execution import configure_execution, deterministic_execution
from .minibatches import backward_minibatches
from .optimizer import MuonWithAdamW
from .run.rng import seed_everything


@dataclass
class TrainingRuntime:
    model: FraiseModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: torch.device
    settings: TrainingSettings
    loss_sum: Callable = token_loss_sum
    compilation_seconds: float = 0.0
    compiled_warmed_up: bool = False
    log: Callable[[str], None] | None = None

    @property
    def use_compile(self) -> bool:
        return self.settings.use_compile

    def step(self, batch):
        with deterministic_execution(self.settings.deterministic):
            return self._step(batch)

    def _compiled_warmup_finished(self, seconds):
        self.compilation_seconds += seconds
        self.compiled_warmed_up = True
        if self.log is not None:
            self.log(
                f"Compiled model forward/backward warmup: {seconds:.3f}s; "
                "reported separately from steady training throughput."
            )

    def _step(self, batch):
        settings = self.settings
        self.optimizer.zero_grad(set_to_none=True)
        warming_up = self.use_compile and not self.compiled_warmed_up
        if warming_up and self.log is not None:
            self.log(
                f"Compiling model and loss for fixed minibatch shape "
                f"[{settings.minibatch_size}, {settings.sequence_length}]."
            )
        loss, tokens = backward_minibatches(
            self.model,
            batch,
            self.device,
            minibatch_size=settings.minibatch_size,
            use_bf16=settings.use_bf16,
            pad_to_minibatch_size=self.use_compile,
            on_first_backward=self._compiled_warmup_finished if warming_up else None,
            loss_sum=self.loss_sum,
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


def resolve_device(device, settings) -> torch.device:
    """Pick the training device and reject unsupported precision early."""
    # Execution settings must precede the BF16 check, which initializes CUDA.
    configure_execution(settings)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if settings.use_bf16 and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("use_bf16 requires a CUDA device with BF16 support")
    return device


def build_runtime(
    config,
    settings,
    steps,
    device: torch.device | None = None,
    *,
    initialize_weights=True,
):
    device = resolve_device(device, settings)
    if device.type == "cuda":
        # TF32 tensor cores for FP32 matmuls; strict mode restores "highest" per step.
        torch.set_float32_matmul_precision("high")
    seed_everything(settings.seed)
    model = FraiseModel(config)
    if initialize_weights and config.tie_word_embeddings:
        # Unit-scale embeddings make the residual stream predict its own token.
        torch.nn.init.normal_(
            model.token_embedding.weight, std=settings.tied_weight_init_std
        )
    model.to(device)
    # Compiling a loss function (not the module) keeps state-dict keys unchanged.
    loss_sum = compile_loss_sum() if settings.use_compile else token_loss_sum
    optimizer = MuonWithAdamW(model, settings)
    scheduler = get_cosine_schedule_with_warmup(optimizer, settings.warmup_steps, steps)
    return TrainingRuntime(model, optimizer, scheduler, device, settings, loss_sum)

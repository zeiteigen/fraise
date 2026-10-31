"""Muon for hidden matrices, with AdamW for embeddings, heads, and vectors."""

import math
from collections import defaultdict

import torch

from fraise.architectures import OptimizerLayout

# Identity recorded in run manifests. Bump it whenever update semantics change so
# runs started under the old rules are not silently resumed under the new ones.
OPTIMIZER_IDENTITY = "muon-adamw-v2"
GROUP_TAGS = ("muon", "adamw", "adamw_no_decay")

# Keller Jordan's quintic Newton-Schulz coefficients, as used by torch.optim.Muon.
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7


def orthogonalize(matrices: torch.Tensor, steps: int) -> torch.Tensor:
    """Batched Newton-Schulz orthogonalization of ``[batch, rows, columns]``.

    This is torch.optim.Muon's iteration with ``baddbmm`` in place of ``addmm``
    so every same-shaped matrix in a step shares one kernel launch per product.
    """
    a, b, c = NS_COEFFICIENTS
    x = matrices.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=NS_EPS)
    for _ in range(steps):
        gram = x @ x.mT
        update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, update, x, beta=a)
    return x.mT if tall else x


def adjusted_learning_rate(lr: float, rows: int, columns: int) -> float:
    """Keller Jordan's original shape scaling (``adjust_lr_fn="original"``)."""
    return lr * math.sqrt(max(1, rows / columns))


class BlockMuon(torch.optim.Optimizer):
    """Muon that treats row blocks of fused matrices as independent matrices.

    A fused Q/K/V projection is three linear maps stored in one weight. Each block
    is orthogonalized and learning-rate scaled on its own shape, exactly as if
    the projections were separate parameters. Same-shaped blocks are batched.
    """

    def __init__(
        self,
        params,
        *,
        lr,
        momentum,
        nesterov,
        ns_steps,
        weight_decay,
        row_blocks=None,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.row_blocks = dict(row_blocks or {})
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters")
                blocks = self.row_blocks.get(parameter, (parameter.size(0),))
                if sum(blocks) != parameter.size(0):
                    raise ValueError("row blocks must partition the matrix rows")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            pending = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter.grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(parameter.grad, 1 - momentum)
                update = (
                    parameter.grad.lerp(buffer, momentum)
                    if group["nesterov"]
                    else buffer
                )
                parameter.mul_(1 - lr * group["weight_decay"])
                start = 0
                for rows in self.row_blocks.get(parameter, (parameter.size(0),)):
                    block = update[start : start + rows]
                    pending[block.shape].append((parameter, start, block))
                    start += rows
            for (rows, columns), blocks in pending.items():
                orthogonal = orthogonalize(
                    torch.stack([block for _, _, block in blocks]), group["ns_steps"]
                )
                step_size = adjusted_learning_rate(lr, rows, columns)
                for (parameter, start, _), update in zip(blocks, orthogonal):
                    parameter[start : start + rows].add_(update, alpha=-step_size)
        return loss


class MuonWithAdamW(torch.optim.Optimizer):
    """Expose both optimizers to a single scheduler and checkpoint store."""

    def __init__(self, model, settings, *, layout: OptimizerLayout | None = None):
        if layout is None:
            from fraise.architectures.transformer import optimizer_layout
            from fraise.model.model import FraiseModel

            if not isinstance(model, FraiseModel):
                raise TypeError("non-transformer models require an optimizer layout")
            layout = optimizer_layout(model)
        hidden_matrices = layout.muon_parameters
        parameters = set(model.parameters())
        if hidden_matrices - parameters:
            raise ValueError("Muon layout contains parameters outside the model")
        if layout.row_blocks.keys() - hidden_matrices:
            raise ValueError("row blocks require a Muon parameter")
        muon_params, decay_params, no_decay_params = [], [], []
        # parameters() deduplicates shared embedding/output weights.
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter in hidden_matrices:
                muon_params.append(parameter)
            elif parameter.ndim >= 2:
                decay_params.append(parameter)
            else:
                # Biases and LayerNorm gains/offsets are not decayed toward zero.
                no_decay_params.append(parameter)
        self.muon = BlockMuon(
            [{"params": muon_params, "optimizer": "muon"}],
            lr=settings.learning_rate,
            momentum=settings.muon_momentum,
            nesterov=settings.muon_nesterov,
            ns_steps=settings.muon_ns_steps,
            weight_decay=settings.weight_decay,
            row_blocks=layout.row_blocks,
        )
        self.adamw = torch.optim.AdamW(
            [
                {"params": decay_params, "optimizer": "adamw"},
                {
                    "params": no_decay_params,
                    "optimizer": "adamw_no_decay",
                    "weight_decay": 0.0,
                },
            ],
            lr=settings.adamw_learning_rate,
            betas=settings.betas,
            weight_decay=settings.weight_decay,
        )
        super().__init__([*self.muon.param_groups, *self.adamw.param_groups], {})
        self._bind_optimizers()

    def _bind_optimizers(self):
        # Loading a state dict replaces groups and state; reconnect both delegates
        # so subsequent scheduler changes and optimizer steps use restored values.
        self.muon.param_groups = self.param_groups[:1]
        self.adamw.param_groups = self.param_groups[1:]
        self.muon.state = self.state
        self.adamw.state = self.state

    def load_state_dict(self, state_dict):
        tags = [group.get("optimizer") for group in state_dict.get("param_groups", [])]
        if tags != list(GROUP_TAGS):
            raise ValueError(
                f"checkpoint optimizer is not {OPTIMIZER_IDENTITY}; start a fresh run"
            )
        super().load_state_dict(state_dict)
        self._bind_optimizers()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

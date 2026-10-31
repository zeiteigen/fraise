"""The configuration, model, and optimizer hooks supplied by an architecture."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Self

import torch

if TYPE_CHECKING:
    from fraise.config.training import TrainingSettings


class ModelConfigProtocol(Protocol):
    @property
    def architecture(self) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_seq_len(self) -> int: ...


class ModelProtocol(Protocol):
    """A PyTorch model whose optional cache can continue an inference sequence."""

    max_seq_len: int
    training: bool

    def forward(
        self, input_ids: torch.Tensor, *, cache: Any = None, last_only: bool = False
    ) -> torch.Tensor: ...

    def __call__(
        self, input_ids: torch.Tensor, *, cache: Any = None, last_only: bool = False
    ) -> torch.Tensor: ...

    def new_cache(self) -> Any | None: ...

    def parameters(self, recurse: bool = True) -> Iterable[torch.nn.Parameter]: ...

    def state_dict(self) -> Mapping[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]): ...

    def to(self, *args, **kwargs) -> Self: ...

    def eval(self) -> Self: ...

    def train(self, mode: bool = True) -> Self: ...


@dataclass(frozen=True)
class OptimizerLayout:
    """Matrices assigned to Muon, including independent fused row blocks."""

    muon_parameters: frozenset[torch.nn.Parameter]
    row_blocks: Mapping[torch.nn.Parameter, tuple[int, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ArchitectureSpec:
    """One registered architecture and its construction and training hooks."""

    id: str
    config_type: type[ModelConfigProtocol]
    build: Callable[[ModelConfigProtocol], ModelProtocol]
    count_parameters: Callable[[ModelConfigProtocol], int]
    initialize: Callable[[ModelProtocol, ModelConfigProtocol, TrainingSettings], None]
    optimizer_layout: Callable[[ModelProtocol], OptimizerLayout]
    label: str
    example_config: Mapping[str, object]
    # Optional: reject an incompatible tokenizer before data preparation.
    check_tokenizer: Callable[[ModelConfigProtocol, Any], None] | None = None
    # Optional: copy tokenizer structure (e.g. macro decompositions) into a model.
    bind_tokenizer: Callable[[ModelProtocol, Any], None] | None = None
    # Optional: named parameter groups that sum to count_parameters(config).
    parameter_breakdown: Callable[[ModelConfigProtocol], Mapping[str, int]] | None = (
        None
    )

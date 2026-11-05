"""Request contracts for the local playground API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelReference(Request):
    run_id: str
    checkpoint_id: str | None = None


class GenerationRequest(Request):
    models: list[ModelReference] = Field(min_length=1, max_length=8)
    prompt: str = Field(min_length=1, max_length=100000)
    max_new_tokens: StrictInt = Field(default=128, ge=0, le=4096)
    temperature: float = Field(default=0.0, ge=0, le=100, allow_inf_nan=False)
    top_k: StrictInt | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    seed: StrictInt = Field(default=1337, ge=0, le=2**63 - 1)


class TrainingRequest(Request):
    # Pydantic reserves model_config for its own configuration.
    configuration: dict[str, Any] = Field(alias="model_config")
    training_config: dict[str, Any]
    tokenizer_id: str
    run_name: str
    resume: ModelReference | None = None


class HierarchySpec(Request):
    """``auto`` shows a macro tokenizer's two-level hierarchy (8k -> 64k)."""

    kind: Literal["auto", "none", "merges", "file", "run"] = "auto"
    level_sizes: list[StrictInt] | None = Field(default=None, max_length=16)
    residual_dims: list[StrictInt] | None = Field(default=None, max_length=16)
    file_id: str | None = None
    run: ModelReference | None = None


class TokenizeConfiguration(Request):
    label: str | None = Field(default=None, max_length=200)
    tokenizer_id: str | None = None
    hierarchy: HierarchySpec = Field(default_factory=HierarchySpec)


class TokenizeRequest(Request):
    text: str = Field(min_length=1, max_length=20000)
    add_special_tokens: bool = False
    configurations: list[TokenizeConfiguration] = Field(min_length=1, max_length=6)

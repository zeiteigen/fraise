# Adding a local architecture

An architecture plugs into the existing causal language-model training and
inference pipeline. Most architectures read and predict the same saved tokenizer
vocabulary. Structured outputs (such as the asymmetric `fraise` architecture,
which predicts base tokens for macro inputs) use the optional hooks described in
[Structured outputs](#structured-outputs).

Keep configuration, model implementation, and adapter hooks in focused modules.
The current adapter, `fraise.architectures.transformer`, is a working reference.

## Configuration and model contract

Use a frozen dataclass with `architecture`, `vocab_size`, and `max_seq_len` fields,
plus architecture-specific settings. Validate its values in `__post_init__`.
The architecture field must identify the registered spec. Configurations are
archived with `dataclasses.asdict`; JSON fields remain flat under their architecture
discriminator. The baseline's existing `ModelConfig` constructor remains available.

The model must be a `torch.nn.Module` exposing `max_seq_len` and:

```python
def forward(self, input_ids, *, cache=None, last_only=False):
    # input_ids: [batch, sequence], integer token IDs
    # Return logits [batch, sequence, vocab_size].
    # With last_only=True, return [batch, 1, vocab_size].
    ...

def new_cache(self):
    # Return None for uncached generation, or a fresh cache whose len() is the
    # number of previously processed tokens and which forward() extends.
    ...
```

Training and scoring call the full forward pass. Generation uses the optional
cache while context fits, then recomputes the most recent `max_seq_len` tokens
when the window slides. Returning `None` always uses full-window recomputation.
Models may opt into `torch.compile` through the existing training settings;
NPU validation requires the model's full forward pass to be exportable by OpenVINO
and fails explicitly if conversion is unsupported.

## Adapter and registration

Define a module-level `ArchitectureSpec` with these hooks:

```python
from fraise.architectures import ArchitectureSpec, OptimizerLayout

def initialize(model, config, settings):
    # Fresh training only; resume and inference restore saved weights.
    pass

def optimizer_layout(model):
    return OptimizerLayout(
        muon_parameters=frozenset({model.hidden.weight}),
        row_blocks={},
    )

SPEC = ArchitectureSpec(
    id="my-model",
    label="My model",
    config_type=MyConfig,
    build=MyModel,
    count_parameters=count_parameters,
    initialize=initialize,
    optimizer_layout=optimizer_layout,
    example_config={
        "architecture": "my-model",
        "vocab_size": 8192,
        "max_seq_len": 512,
        # Supply every required architecture-specific field.
    },
)
```

`count_parameters(config)` must count unique trainable weights without allocating
the full model. `optimizer_layout` identifies hidden 2D matrices for Muon; all
remaining matrices use AdamW, while biases and other vectors use AdamW without
decay. Use an empty Muon set when appropriate. Shared parameters are deduplicated.
For fused matrices, `row_blocks` maps a Muon parameter to row counts that partition
it into independent updates, as the Transformer adapter does for Q/K/V.

Import and register the spec at module scope in `fraise.architectures.__init__`,
after the registry exports:

```python
from .my_architecture import SPEC

register_architecture(SPEC)
```

This central import makes registration available to CLI processes, the web server,
and newly spawned training workers. Registering only inside a notebook session
also works for Python training and checkpoint validation; that session's registry
changes are not inherited by independently launched CLI/web training processes.
Config classes, model builders, and hook functions must be importable at module
scope so validation workers can unpickle them under multiprocessing `spawn`.

New models automatically appear in the web architecture selector. `example_config`
provides the initial settings; scalar dataclass fields become form controls and
the advanced JSON editor exposes the full configuration.

## Structured outputs

Models whose forward pass does not return vocabulary logits can still use the
shared training, validation, and inference paths:

- `target_loss_sum(input_ids, targets, target_mask)` on the model returns the
  summed negative log-likelihood of the valid targets. When present, training,
  validation, and `score_text` call it instead of applying cross-entropy to
  `forward`'s output.
- `hierarchical = True` on the model routes `iter_generate`, `top_next_tokens`,
  and `benchmark_generation` to trie-constrained base-token generation
  (`fraise.inference.hierarchical`). It needs a `local` decoder, an `end_id`,
  and `decompositions`/`macro_lengths` buffers; see `FraiseHierarchicalModel`.

Three optional `ArchitectureSpec` fields support tokenizer-dependent models:

```python
SPEC = ArchitectureSpec(
    ...,
    check_tokenizer=check_tokenizer,  # (config, tokenizer): raise ValueError early
    bind_tokenizer=bind_tokenizer,    # (model, tokenizer): copy structure into buffers
    parameter_breakdown=breakdown,    # (config) -> {group: count}, summing to the total
)
```

`check_tokenizer` runs with the other compatibility checks, before data
preparation. `bind_tokenizer` runs on every run start, before weights are
restored; store what it copies in persistent buffers so checkpoints are
self-contained. `params` prints the breakdown under the total.

## Validation

Use a tiny model and local text fixtures to verify parameter counts, optimizer
coverage, a training step, checkpoint/final-weight round trips, exact resume,
generation with and without caching, and spawned validation. Existing tests in
`tests/test_architectures.py` demonstrate the full path using a test-only model.

For the CPU regression suite, including the web API:

```sh
uv run --extra web python -m unittest discover -s tests -v
```

When working in a restricted environment, point `HF_DATASETS_CACHE` and
`UV_CACHE_DIR` at writable cache directories. The web API tests require normal
local event-loop sockets even though their model/data fixtures are offline.

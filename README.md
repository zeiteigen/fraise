# Fraise

Fraise explores language models with a large macro input vocabulary and a smaller
base output vocabulary, as described in [PoC.md](PoC.md). Implemented stages:

1. Baselines `base-8k` and `base-64k`: the `transformer` architecture with an 8k or a
   64k byte-level BPE tokenizer.
2. Hierarchical-only 64k (`hier-64k`): the same `transformer` with the 64k hierarchical
   (macro) tokenizer and an independent embedding row per macro.
3. Fraise 64/8 (`fraise-64-8`): the `fraise` architecture, compositional 64k macro input
   and 8k base-token output.
4. Compositional macro embeddings (`macro_embedding`, ablatable to `independent`).
5. Learned 32-dimensional macro residuals (`macro_residual_dim`; 0 disables them).
6. Trie-based autoregressive bridging between the global and local decoders.
7. Multi-level hierarchical input embeddings with per-token residual widths
   ([Hierarchical embeddings](#hierarchical-embeddings-8k--16k--64k--)).

Multi-token prediction, dynamic decomposition, and variable-resolution KV caches
remain experiment stages. See [Asymmetric Fraise](#asymmetric-fraise-macro-in-base-out).

## Model workspace (AblationKit)

The interactive views run inside [AblationKit](.ablationkit/) as project panels, through
fraise's own code in `fraise.workbench` (the project environment, at the checkout's current
commit; commit changes to `fraise.workbench` before rebuilding an extension):

- **Autocomplete** (`fraise-workspace`): edit a document and generate from the text before the
  cursor. Review the streamed continuation, then accept it at the saved cursor position or
  discard it. Accepting replaces any selected text and preserves the rest of the document.
  Ctrl/Cmd+Enter generates, Tab accepts, and Escape discards. The model receives raw text
  without roles or chat templates. Inputs, checkpoint choices, and sampling settings are kept in
  `DATA_DIR/workspace/saved-inputs.json`. The inspector shows model dimensions, vocabulary,
  context limit, checkpoint step, and validation; generation reports speed and token counts.
- **Compare checkpoints** (`fraise-workspace`): send the same prompt, sampling settings, and seed
  to up to eight checkpoints, one after another. Outputs include generation speed, loading time,
  saved settings, tokenizer identity, and available validation results. Validation loss depends
  on tokenization and is not a direct comparison across different tokenizers.
- **Train** (`fraise-workspace`): select a tokenizer, start from an architecture preset or a
  saved run's configuration, edit forms or complete JSON, check it (parameter count), or resume a
  checkpoint (only operational settings change). **Prepare** writes a checked plan to
  `DATA_DIR/launch/<run>/<time>/`.
- **Tokenization workbench** (`tokenization-workbench`): inspect how text is segmented and how
  each input token is composed through an embedding hierarchy; compare several configurations.

Training runs as an AblationKit job from the `MLproject` entrypoints (Runs → More →
Entrypoints): `launch` takes a plan directory from the Train panel, and `train` takes
configuration files directly. The job page shows the console; the run page shows loss,
throughput, and validation by global step (`train --ablationkit-metrics` logs them through the
`ablationkit_sdk` that AblationKit puts on every job's path). Cancelling the job, or reaching its
time limit, sends SIGTERM: `train` saves a resumable checkpoint after the current optimizer step
and exits, so a stopped run remains unfinished and can be resumed from the Train panel. Without
`--stop-file`, Ctrl+C still interrupts as usual.

The panels read runs from `DATA_DIR/models` and tokenizers from `DATA_DIR/tokenizers`. Generation
uses CUDA when available and keeps the last checkpoint loaded until another is needed, the
extension's worker idles out, or **Release device memory** is used; nothing coordinates it with
running training jobs, so release it before training on the same GPU. See each extension's
README for details.

## Model architectures

Model configuration JSON names its architecture: `"architecture": "transformer"` or `"fraise"`
(required). Architecture selection is shared by training, validation, inference, and parameter
counting.

The registry in `fraise.architectures` supplies configuration, construction,
initialization, parameter-counting, and optimizer hooks. See
[Adding an architecture](docs/architectures.md) for the contract and registration
workflow. The built-in registry provides the Transformer and the asymmetric
Fraise model.

## Setup

```sh
uv sync
```

The application is packaged under `fraise`. Commands use `./data` as the shared
data root. Pass `--data-dir /path/to/data` or set `FRAISE_DATA_DIR` to use another
location, including when running outside the repository. `--save-path` can
separately override the parent directory of model runs.

## Create a tokenizer

Select the sources explicitly. Sources stream from the Hugging Face Hub with a
0.1 GB download cap each by default. Optionally cap
text consumed during tokenizer training with `--max-bytes-per-source`. Limits
keep complete documents and can exceed the requested size by one document.
FineWeb's tokenizer text starts after the 0.05 GB prefix reserved for validation,
so validation documents never shape the tokenizer's merges; `--source-size-gb`
still measures the text available to the tokenizer.

```sh
uv run create-tokenizer \
  --vocab-size 8192 \
  --sources cosmopedia-khanacademy cosmopedia-stories fineweb-edu \
  --source-size-gb 0.1 \
  --max-bytes-per-source 100000000
```

The command prints the tokenizer JSON path. Tokenizers and `metadata.json` live
under `DATA_DIR/tokenizers`. Artifact filenames include the training request
fingerprint, so creating another tokenizer preserves existing artifacts. A
matching request reuses a tokenizer only when its checksum matches.

Loading a tokenizer preserves its saved vocabulary and post-processing. It must
already contain `<|pad|>`, `<|bos|>`, and `<|eos|>`. Newly trained tokenizers add BOS
and EOS once per document. Byte-level tokenizers require at least 259 vocabulary
entries for the byte alphabet and special tokens.

## Create a macro tokenizer

A macro tokenizer learns merges over the *IDs* of an existing base tokenizer, so
every macro has one deterministic decomposition into base tokens:

```sh
uv run create-tokenizer \
  --base-tokenizer data/tokenizers/fraise-8192-....json \
  --vocab-size 65536 \
  --max-macro-length 8 \
  --sources fineweb-edu \
  --max-bytes-per-source 200000000
```

Text is encoded by the base tokenizer, each base ID becomes one private-use
character, and the Rust BPE trainer merges adjacent IDs. Macro IDs below the base
vocabulary size are the base tokens themselves; merge `k` creates macro
`base_vocab_size + k`. Special tokens never merge, a base token containing a line
break ends a macro, and no macro exceeds `--max-macro-length` base tokens.
Expanding a macro encoding always reproduces the base encoding exactly.

The artifact is one JSON file (`fraise-macro-VOCAB-KEY.json`) that embeds the
base tokenizer and its checksum, so runs archive it like any tokenizer and
`load_tokenizer` recognizes it. It supplies `encode`, `encode_batch`, `decode`,
`expand`, and `decomposition`. Merging over 100 MB of text takes about two minutes.

### Budgets count base tokens

PoC comparisons need every architecture to see the same text, so token budgets
(`dataset_token_counts`, `max_train_tokens`, `max_validation_tokens`) always count
base tokens. With a macro tokenizer a target macro counts as its decomposition
length. Downloads are selected with the base tokenizer, so a macro run and an 8k
run with the same budgets reuse the same raw cache. Each source ends with the
macro that reaches its count, so a source can exceed its count by less than one
macro; planning computes these exact counts (logged at the start) and checks them
at the end of training. Progress, throughput, and `tokens_seen` report base
tokens, while training loss stays the mean over the model's own targets.

### Bits per byte

Validation reports bits per byte as well as loss whenever the base tokenizer is
byte-level BPE: summed negative log-likelihood in bits divided by the UTF-8 bytes
of the scored targets. BPB compares models with different tokenizers, which
token losses do not. It appears in the log, in `validation_results`, in the workspace
inspector, and in `score_text`.

## Configure and train

Training takes explicit JSON configuration and tokenizer paths. Commands do not
prompt for missing arguments. JSON and Python callers use the same normalized
configuration classes and defaults.

Example model configuration:

```json
{
  "vocab_size": 8192,
  "residual_stream_dim": 512,
  "head_dim": 64,
  "qk_dim": 64,
  "num_heads": 8,
  "swiglu_up_dim": 1536,
  "num_layers": 8,
  "max_seq_len": 512,
  "tie_word_embeddings": false
}
```

`max_seq_len` defaults to 2048 when omitted. `tie_word_embeddings` defaults to
false; when true the embedding and output weights are shared and the output
projection keeps its bias. `params MODEL_CONFIG.json` reports the corresponding
parameter count.

The checked-in `hyperparameters.example.json` provides the training configuration:

```json
{
  "batch_size": 64,
  "minibatch_size": 8,
  "learning_rate": 0.02,
  "adamw_learning_rate": 0.0003,
  "muon_momentum": 0.95,
  "muon_nesterov": true,
  "muon_ns_steps": 5,
  "betas": [0.9, 0.95],
  "weight_decay": 0.01,
  "sequence_length": 512,
  "max_train_tokens": 250000000,
  "dataset_token_counts": {
    "fineweb-edu": 250000000,
    "cosmopedia-khanacademy": 0,
    "cosmopedia-stories": 0
  },
  "max_validation_tokens": 32768,
  "validation_timeout_seconds": 300,
  "validation_cpu_threads": 4,
  "validation_device": "cpu",
  "shuffle_data": false,
  "use_bf16": true,
  "use_compile": true,
  "compile_threads": 4,
  "deterministic": false,
  "log_interval_steps": 100,
  "tied_weight_init_std": 0.02
}
```

The required training fields are `batch_size`, `learning_rate`,
`max_train_tokens`, and `dataset_token_counts`. Dataset counts are non-negative
integers, include every registered source, and sum exactly to `max_train_tokens`.
Training makes a single pass: each source supplies its count of targets once, never
repeating data, and the learning-rate schedule covers that pass. The pass ends with a
loss report and a validation checkpoint.
Sources assigned zero tokens are skipped. Counts measure causal targets produced
by the selected tokenizer, rather than words or an estimated byte/token ratio.

`minibatch_size` defaults to `batch_size` and must divide it; gradient accumulation
steps are derived as `batch_size / minibatch_size`. Gradients are weighted by valid targets, including padded tails.

Other defaults are defined in `fraise.config.training.TrainingSettings`:
`adamw_learning_rate=0.0003`, `muon_momentum=0.95`, `muon_nesterov=true`,
`muon_ns_steps=5`, `betas=[0.9,0.95]`, `weight_decay=0`, `warmup_steps=0`, `sequence_length=512`,
`packing_stride=sequence_length`, `seed=1337`, `checkpoint_interval_steps=1000`,
`checkpoint_keep_last=2`, `tokenization_batch_size=1000`, `use_bf16=false`,
`use_compile=true`, `compile_threads=4`, `deterministic=false`, `log_interval_steps=100`,
`shuffle_data=false`, `gradient_clip_norm=1`, `max_validation_tokens=32768`,
`validation_timeout_seconds=300`, `validation_cpu_threads=4`, and
`validation_device="cpu"`. The checked-in config selects `"cpu"`. Set
`max_validation_tokens` in `hyperparameters.json` to change the number of causal
validation targets per checkpoint. Set it to `0` to disable validation: held-out
data loading, tokenization, and CPU/NPU validation workers are skipped. `null`
evaluates the complete held-out split; positive integers set a target limit.
BF16 training requires a supporting CUDA device. CPU validation uses FP32.
CPU validation is a good starting point for small models and validation budgets;
NPU conversion and compilation are repeated for each checkpoint and can dominate
short evaluations. Benchmark both backends for larger workloads. Increase
`checkpoint_interval_steps` when validation queue waits dominate training.

Training uses Muon for the transformer's hidden 2D matrices, with the
`original` learning-rate scaling of PyTorch's Muon. The fused query/key/value
projection is treated as three matrices: Q, K, and V are orthogonalized and
learning-rate scaled separately, as if they were separate parameters. Same-shaped
matrices are orthogonalized together in batched matmuls. Embeddings and output
projection weights use AdamW with `weight_decay`; biases and normalization
parameters use AdamW without weight decay. Shared embedding/output weights are
updated once. `learning_rate` controls Muon; `adamw_learning_rate` and `betas`
control AdamW. Both optimizers share the cosine schedule with `warmup_steps`. The example starts at Muon LR 0.02 and AdamW LR 0.0003,
with weight decay 0.01, following the [Muon usage guidance](https://github.com/KellerJordan/Muon).
Learning rate and weight decay still need tuning for the training run.

`tied_weight_init_std` is a training hyperparameter that defaults to 0.02 and
must be a finite positive number. For a fresh run with `tie_word_embeddings`
enabled, training initializes the shared weights from a zero-mean normal
distribution with this standard deviation. Untied models keep their existing
initialization. The value is archived in `training_hyperparameters.json` and
the run manifest. Checkpoint loading and resumed training restore saved weights
without applying this initialization; changing initialization requires a fresh run.

For NPU validation, install OpenVINO and the Intel NPU user-space driver:

```sh
uv sync --extra npu
bash scripts/install_npu_runtime.sh
```

The installer requires `curl` and `bsdtar` (libarchive). It downloads pinned,
SHA-256-checked official Intel driver and Level Zero packages to
`~/.local/share/fraise/npu-runtime/lib`; it does not install firmware or change
kernel/system configuration. The backend loads these libraries in its worker.
A system-installed Intel NPU runtime also works. The Linux kernel driver must
already expose an accessible `/dev/accel` device. See
[Intel's driver releases](https://github.com/intel/linux-npu-driver/releases/tag/v1.38.0).

`validation_device="npu"` converts each checkpoint with `torch.export` to a
static OpenVINO graph and explicitly compiles for `NPU`. Forward passes use the
NPU's floating-point precision, typically FP16, with CPU FP32 loss reduction;
validation losses can differ slightly from CPU FP32 results. One sequence is
processed per NPU request, independently of the training minibatch size. Padded
targets remain excluded from the loss. Logs verify `EXECUTION_DEVICES` and
include conversion/compilation progress. Compilation is included in the timeout.
There is no silent CPU fallback: discovery, conversion, or compilation failures
are recorded as validation failures. Select `"cpu"` explicitly to use CPU
validation. OpenVINO's NPU device API is documented
[here](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html).

Hardware regression tests can be run explicitly:

```sh
FRAISE_TEST_NPU=1 uv run --extra npu python -m unittest discover -s tests -p 'test_validation_npu.py' -v
```

```sh
uv run --extra npu train \
  --training-hyperparameters hyperparameters.json \
  --model-config configs/model_config.json \
  --tokenizer-file /path/to/tokenizer.json \
  --model-name baseline-001
```

`use_compile=true` compiles the forward pass and the masked token loss together
as one fixed-shape graph; the module itself is not wrapped, so checkpoint keys are
unchanged. Masked targets are ignored by the loss rather than gathered out, which
avoids a device sync per minibatch. Compiled training pads short
minibatches to `minibatch_size`, masking all added rows out of the loss and token
counts. This keeps the model's input shape fixed through source tails. Set
`use_compile=false` for small smoke tests where compilation costs more than it
saves; eager execution keeps short minibatches at their original size.
`compile_threads` controls TorchInductor's compiler pool (default four workers)
and takes precedence over its environment-derived worker count for this run.
Compiler pools are process-wide; use a fresh Python process when changing their size.
The first compiled forward/backward warmup is timed and logged separately; it
includes the first real minibatch execution, rather than an isolated compiler
measurement.

Non-deterministic CUDA runs use TF32 for FP32 matmuls
(`float32_matmul_precision="high"`); the manifest records the precision in effect.

`deterministic=true` enables strict Torch deterministic algorithms, forces the
math attention backend and highest FP32 matmul precision, and disables cuDNN
benchmarking during training and validation. It sets `CUBLAS_WORKSPACE_CONFIG`
to `:4096:8` before initializing training CUDA when the variable is unset.
Strict mode can cost throughput; it is intended for repeatable experiments and
resume verification on the same hardware/software setup. Torch execution flags
are restored after each step or evaluation, including on failure. The selected
attention policy and execution settings are recorded in the run manifest.

`log_interval_steps` controls progress reports independently of checkpointing.
Reports include both optimizers' next learning rates, loss, gradient norm,
tokens/second, validation queue wait time, and allocated/reserved/peak CUDA memory.
Throughput excludes the initial compiled warmup and includes queue waits and
checkpoint overhead. The end of training also reports progress. The queue logs each
backpressure wait and its running total.

Passages within each source are read in their stored order. `shuffle_data` randomly interleaves packed windows across all active sources
before batching, while preserving each source's exact token allocation. Mixing
uses bounded memory and samples sources according to their remaining window
counts. Mixing is reproducible from `seed` and is shared by training and preview.

When `shuffle_data=false`, sources are traversed in the fixed order
FineWeb-Edu, Cosmopedia Khan Academy, and Cosmopedia stories.

Documents within each source form a continuous causal stream before packing;
BOS/EOS mark their boundaries without blocking
attention between documents. A stride below the context length repeats targets
across overlapping windows, and those repeats count toward the allocation.

Each source must supply its allocated targets without repeating data; a capacity
check rejects larger allocations. FineWeb
reserves the first 0.05 GB of source text for validation even when its training
allocation is zero.

## Preview the training data

```sh
uv run visualize-training-data \
  --training-hyperparameters hyperparameters.json \
  --model-config configs/model_config.json \
  --tokenizer-file /path/to/tokenizer.json \
  --count 100 \
  --first-source-only \
  --output data/training_examples.html
```

Preview shares training's tokenization, token plan, seed, stride, padding, and
batch iterator. Source labels travel with the packed windows. It skips held-out
validation downloads. `--first-source-only` avoids later sources when the first
source supplies every requested window; otherwise omit the flag.

## Runs, checkpoints, and inference

Runs are stored under `DATA_DIR/models/MODEL_NAME`. Fresh runs refuse to overwrite
a non-empty directory. Each run contains canonical `model_config.json` and
`training_hyperparameters.json` archives, `tokenizer.json`, `model_meta.json`, an
append-only `training.log`, step checkpoints, and the final `MODEL_NAME.pt`.
Supplied configuration bytes are archived under their role-based filenames.

```sh
uv run --extra npu train \
  --training-hyperparameters hyperparameters.json \
  --model-config configs/model_config.json \
  --tokenizer-file /path/to/tokenizer.json \
  --resume data/models/baseline-001
```

When resuming, the model name comes from the run unless explicitly supplied.
The latest numerically ordered step checkpoint is used, or pass a checkpoint
file explicitly. Resume compares model, training, dataset, and tokenizer identity.
Python/Torch versions, execution device, and Git provenance are recorded
separately, so environment changes alone do not invalidate continuation.
Checkpoints restore both optimizers and their learning rates. The optimizer
identity is `muon-adamw-v2` (separate Q/K/V orthogonalization and no decay on
biases and normalization parameters). Runs started with AdamW or the earlier
`muon-adamw` rules require a fresh run; their model weights remain loadable for
inference and export.
Operational settings may change on resume: `checkpoint_interval_steps`,
`checkpoint_keep_last`, `compile_threads`, `log_interval_steps`,
`tokenization_batch_size`, `validation_cpu_threads`, `validation_device`, and
`validation_timeout_seconds`. Each resume records their values in its provenance.
All other settings are compared after filling defaults for fields that older
manifests lack. Changing `use_compile` or `deterministic` requires a fresh run
because it can change training numerics. The run directory and resume identity
are checked before tokenization or compilation begins.

Resuming from a checkpoint older than the run's newest one starts a new
continuation. The later checkpoints from the abandoned continuation are moved
(never deleted) into `superseded-after-step-NNNNNNNN/` inside the run, so
retention and directory resumes only see the live continuation.

Training retains the newest two checkpoints by default. Active and pending
validation checkpoints remain protected until evaluation finishes. Every
checkpoint receives a validation attempt. The queue holds one active checkpoint
and one waiting checkpoint; when both slots are occupied, training waits before
queuing another. This bounds disk usage without silently skipping checkpoints.

`training.log` reports submissions, queued work, worker startup and PID,
checkpoint loading, token progress, throughput, losses, errors, and timeouts.
Worker messages are relayed independently of training steps. Each checkpoint's
`validation_timeout_seconds` deadline includes worker initialization, checkpoint
loading, and evaluation. A watchdog kills an overdue worker even while the
training thread is busy; the next checkpoint starts a fresh worker. Exiting
training allows idle workers two seconds to exit cleanly, then kills them if
necessary. Interruption kills an active worker immediately. On Linux,
workers also die automatically if their training parent disappears.

Metadata records the latest validation result, failures, and the
`validation_results` history during training, including attempt status and elapsed
time. Resume preserves previous attempts. Checkpoints include optimizer,
scheduler, counters, and RNG state; the final model contains weights.
Checkpoints and weights load with `torch.load(weights_only=True)`, so no pickled
code is executed. Schema-4 checkpoints store NumPy's RNG state as tensors; older
schema-2/3 checkpoints load with an allowlist of the NumPy array types they contain.

```sh
uv run checkpoint-to-model CHECKPOINT.pt OUTPUT.pt
```

Reusable inference functions live in `fraise.inference`; the workspace panels
call the same run loader and generation functions. Run loading preserves weight tying and the archived tokenizer.
Generation and the benchmark decode with a key/value cache and project only
the last position onto the vocabulary. Once a sequence exceeds `max_seq_len`, the
context window slides and each step recomputes the visible window. Positive
sampling temperatures work with the default nucleus sampler; optional top-k
filtering can also be supplied.

## Asymmetric Fraise (macro in, base out)

`model_config.fraise.example.json` configures the `fraise` architecture. Train it
with a macro tokenizer whose vocabulary sizes match `vocab_size` (macro input)
and `base_vocab_size` (base output); `max_macro_length` must cover the
tokenizer's longest macro:

```sh
uv run params model_config.fraise.example.json
uv run train \
  --training-hyperparameters hyperparameters.json \
  --model-config model_config.fraise.example.json \
  --tokenizer-file data/tokenizers/fraise-macro-65536-....json \
  --model-name fraise-001
```

**Macro inputs.** With `"macro_embedding": "compositional"`, the input for macro
`m` with base decomposition `b_1 .. b_k` is

    sum_i g_i * E[b_i] / sqrt(k)  +  length(k)  +  P r(m)

with per-position gates `g_i`, a shared base embedding table `E`, and a
`macro_residual_dim`-dimensional residual `r(m)` projected to the model width.
The projection starts at zero, so every macro begins as its constituents; weight
decay pulls rarely seen residuals back toward the shared composition.
`"independent"` (with `macro_residual_dim: 0`) uses one full embedding per macro
for the ablation. `params` reports the dense transformer separately from lexical
tables, including the per-macro (`macro_lexical`) parameters.

**Global transformer.** The same transformer as the baseline runs over macro
positions, so `sequence_length` and `max_seq_len` count macros and the KV cache
holds one entry per macro.

**Local decoder and trie bridging.** For the macro after position `t`, a small
decoder emits its base tokens and then an END symbol, over an output vocabulary
of `base_vocab_size + 1`. Its state for slot `j` is a projection of the global
hidden state plus a gated sum of the base tokens already emitted for that macro,
followed by `local_num_layers` SwiGLU blocks of width `local_dim`. Every slot
depends only on its own context and prefix, so training evaluates all real slots
in one batch. A macro's loss is the summed cross-entropy of its base tokens and
END, so training and validation losses are negative log-likelihoods per macro,
directly comparable to Hierarchical-only 64k (`hier-64k`) on the same tokenizer.

Generation walks a trie of macro decompositions. Each base token is sampled from
the children of the current trie node, plus END when the path already spells a
complete macro (sampling settings apply within that set); a leaf ends the macro
without consulting the decoder. Only complete macros reach the global transformer,
whose KV cache advances once per macro. The last prompt macro is re-decided: it is
removed from the global context and its base tokens are forced as the first trie
path, so a prompt ending in `straw` can continue into `strawberry`. Generated
`ids` and token counts are base tokens. `benchmark_generation` also reports
global transformer steps per generated token.

Macro decompositions are persistent model buffers, bound from the tokenizer when
a run starts, so checkpoints and final weights are self-contained. The variable
number of local slots is kept out of the fixed-shape compiled graph; the rest of
the forward pass compiles normally. NPU validation is not supported for
hierarchical models; use `validation_device: "cpu"`.

## Hierarchical embeddings (8k → 16k → 64k → …)

`"macro_embedding": "hierarchical"` generalizes the compositional 8k → 64k input
embedding to any number of levels, such as 8k → 16k → 64k or 8k → 32k → 128k → 1M.
Only the final input tokenizer segments text and creates transformer positions;
intermediate levels are representation layers that build each input token's
embedding. The base output vocabulary, local decoder, trie, and generation are
unchanged, and existing configurations keep the two-level `MacroEmbedding`.

**Model.** Vocabularies are nested: level `l` contains nodes `0 .. vocab_size_l - 1`,
level 0 is the atom (base-token) table, and the top level contains every input
token. At each level `l >= 1`, every node in the level's vocabulary is

    x_l(n) = sum_i g_i * x_{l-1}(c_i) / sqrt(k) + length_l(k)  [+ P_l r(n)]

where `c_1 .. c_k` are its ordered children (a node carried over from level `l-1`
is composed from itself). Each node owns **one** residual `r(n)`, added at the
level that introduces it (atoms at level 1), so a token present at several levels
never gets duplicate residual parameters. Residual widths are per token and may
be 0; values are packed per level and a width-`w` residual uses the first `w` rows
of the level's zero-initialized projection. `composition` is `"per_level"` (gates,
length embeddings, and projection per level) or `"shared"`. With two levels this
reproduces the original compositional formula exactly. Forward passes compose each
distinct node of a batch once per level and gather the results back to positions;
this step runs eagerly (its sizes depend on the batch), while the transformer
still compiles.

**Configuration.** Level shapes live in the model configuration, so `params`
counts parameters exactly and reports residual memory per level:

```json
"macro_embedding": "hierarchical",
"macro_residual_dim": 0,
"embedding_hierarchy": {
  "levels": [
    {"vocab_size": 8192},
    {"vocab_size": 16384, "max_composition_length": 8, "residual_dim": 16},
    {"vocab_size": 65536, "max_composition_length": 8, "residual_dim": 32}
  ],
  "composition": "per_level",
  "source": "tokenizer_merges"
}
```

`"source": "tokenizer_merges"` derives the structure from the macro tokenizer:
intermediate levels are merge-order prefixes, and a macro's children are found by
expanding its merge tree down to the level below. It is a simple valid hierarchy
for experiments, not a vocabulary-selection algorithm. Residual widths are uniform
per level (level 1's width also applies to atoms). Changing `levels` is all it
takes to train a different hierarchy over the same tokenizer and data caches.

For an externally generated hierarchy use `"source": "file"` with `"path"` and the
file's `"sha256"`. Non-uniform levels also state `"residual_parameters"` (the sum
of their owned widths), so parameter shapes are known before the file is read:

```sh
uv run fraise-hierarchy build --tokenizer-file data/tokenizers/fraise-macro-65536-....json \
  --level-sizes "8192 16384 65536" --residual-dims "16 32" \
  --output data/hierarchies/merges-8k-16k-64k.npz   # prints the config block
uv run fraise-hierarchy inspect data/hierarchies/merges-8k-16k-64k.npz \
  --tokenizer-file data/tokenizers/fraise-macro-65536-....json
uv run fraise-hierarchy show "The strawberry field" \
  --tokenizer-file data/tokenizers/fraise-macro-65536-....json \
  --hierarchy data/hierarchies/merges-8k-16k-64k.npz
```

At run start (fresh or resumed) the definition is checked against the
configuration (level sizes, composition lengths, residual widths and totals) and
against the tokenizer: every input token must expand to exactly its tokenizer
decomposition. It is then copied into persistent buffers, so checkpoints and final
weights are self-contained: inference, export, and the visualizer read the
structure from the weights. Resuming re-reads the file, so keep it at its
configured path (relative paths resolve from the working directory). The packed
residuals use AdamW with weight decay like the two-level residual table.

**Hierarchy file format** (`fraise.hierarchy.EmbeddingHierarchy`, `.npz` or `.json`):

| Field | Shape | Meaning |
|---|---|---|
| `level_sizes` | `[L]` | Cumulative nested node counts; `level_sizes[0]` is the base vocabulary. |
| `compositions_l` (npz) / `compositions[l-1]` (json) | `[new_l, K_l]` | Ordered children (nodes `< level_sizes[l-1]`) of each new node `level_sizes[l-1] + i`; `-1` pads rows (json: ragged lists). |
| `residual_widths` | `[level_sizes[-1]]` | One width per node, `>= 0`; owned by the introducing level (atoms by level 1). |
| `input_nodes` | `[input_vocab]`, optional | Node of each input token; omit for the identity. Lets intermediate nodes be tokens the input tokenizer never emits. |
| `header` (npz) | JSON | `format: "fraise-embedding-hierarchy"`, `format_version: 1`, `level_sizes`, optional `level_names` and `metadata`. |

`EmbeddingHierarchy(level_sizes, compositions, residual_widths, input_nodes).save(path)`
validates and writes either format and returns the sha256; `config_levels()`
returns the matching `levels` list.

**Visualizer.** In AblationKit, open the project's **Tokenization workbench** panel. Enter any text and add up to six
configurations to compare: a macro tokenizer's default two-level hierarchy,
merge-prefix levels typed as sizes, a file in `DATA_DIR/hierarchies/`, or the
hierarchy stored in a saved run's checkpoint. Each result separates **text
segmentation** (input tokens = transformer positions, and base output tokens) from
**representation composition**: a level-by-level icicle in which each cell spans
the base tokens it covers (dashed cells recompose carried nodes from themselves), a level table
with residual ownership and parameters, expandable composition trees, and a node
inspector with IDs, text, levels, residual width, and children.

## Existing artifacts

Saved runs need a manifest (version 2 or 3) that names their architecture, with schema 3,
4, or 5 checkpoints; all of these load for inference. Resuming needs a version-3 manifest
(single-pass training) with schema-5 checkpoints and a matching optimizer; runs from
before single-pass training are finished and cannot be resumed. Tokenizer
JSON files load as is when they contain the required special tokens, and tokenizer
directories keep a schema-3 `metadata.json` index. Older formats (version-1 manifests,
schema-2 checkpoints and their former weight names, the `shuffle`,
`shuffle_per_epoch`, `num_epochs`, and `gradient_accumulation_steps` settings, the old
tokenizer index) are no longer read; their
migrations have been removed.

## Storage and development

Raw streaming downloads have request-specific directories under
`DATA_DIR/raw/SOURCE/raw/`. Validation uses its own request cache and survives
changes to training allocations. Arrow caches use local hard links when possible;
source changes invalidate them. Tokenized prefixes are memory-mapped int32 token
caches on disk, reused across runs and resumes; the first run after upgrading
from int64 caches rebuilds them once. Packing reads Arrow buffers directly as NumPy
arrays, and a resumed run jumps to its next window without re-packing earlier
batches. Removing a cache causes it to be rebuilt.

The package separates model components, configuration, source acquisition,
tokenizer creation, training, and inference. Within `fraise.training`,
`pipeline/` tokenizes, packs, mixes, and batches data; `run/` owns run directories,
checkpoints, manifests, and resumable state; and `validation/` runs the isolated
checkpoint evaluator.
Small modules own their concrete behavior; package initializers stay lightweight.

```sh
uv run python -m unittest discover -s tests -v
```

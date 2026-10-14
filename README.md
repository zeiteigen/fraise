# Fraise

Fraise explores language models with a large macro input vocabulary and a smaller
base output vocabulary. The current implementation is the conventional
byte-level BPE transformer baseline described in [PoC.md](PoC.md). Hierarchical
composition, trie decoding, and dynamic decomposition remain experiment stages.

## Setup

```sh
uv sync
# For the notebooks:
uv sync --extra notebooks
```

The application is packaged under `fraise`. Commands use `./data` as the shared
data root. Pass `--data-dir /path/to/data` or set `FRAISE_DATA_DIR` to use another
location, including when running outside the repository. `--save-path` can
separately override the parent directory of model runs.

## Create a tokenizer

Select the sources explicitly. Streaming sources default to a 0.1 GB download
cap each; BabyLM uses its complete pinned native text snapshot. Optionally cap
text consumed during tokenizer training with `--max-bytes-per-source`. Limits
keep complete documents and can exceed the requested size by one document.
FineWeb's tokenizer text starts after the 0.05 GB prefix reserved for validation,
so validation documents never shape the tokenizer's merges; `--source-size-gb`
still measures the text available to the tokenizer.

```sh
uv run create-tokenizer \
  --vocab-size 8192 \
  --sources babylm cosmopedia-khanacademy cosmopedia-stories fineweb-edu \
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
  "num_epochs": 1,
  "sequence_length": 512,
  "max_train_tokens": 250000000,
  "dataset_token_counts": {
    "babylm": 100000000,
    "fineweb-edu": 150000000,
    "cosmopedia-khanacademy": 0,
    "cosmopedia-stories": 0
  },
  "max_validation_tokens": 32768,
  "validation_timeout_seconds": 300,
  "validation_cpu_threads": 4,
  "validation_device": "cpu",
  "shuffle_per_epoch": false,
  "shuffle_data": false,
  "use_bf16": true,
  "use_compile": true,
  "compile_threads": 4,
  "deterministic": false,
  "log_interval_steps": 100,
  "tied_weight_init_std": 0.02
}
```

The required training fields are `batch_size`, `learning_rate`, `num_epochs`,
`max_train_tokens`, and `dataset_token_counts`. Dataset counts are non-negative
integers, include every registered source, and sum exactly to `max_train_tokens`.
Both `max_train_tokens` and the source counts apply to each epoch. Every epoch
reuses the selected tokenized corpus with the same source allocations; total
training exposure is `max_train_tokens * num_epochs`. For example, 150,000
targets and three epochs produce 450,000 training targets, with a loss report
and validation checkpoint at the end of each epoch. The learning-rate schedule
covers all epochs.
Sources assigned zero tokens are skipped. Counts measure causal targets produced
by the selected tokenizer, rather than words or an estimated byte/token ratio.

`minibatch_size` defaults to `batch_size`. For existing configs,
`gradient_accumulation_steps` may be supplied instead; it is normalized to a
minibatch size and then derived for reporting. Their product must equal
`batch_size`. Gradients are weighted by valid targets, including padded tails.

Other defaults are defined in `fraise.config.training.TrainingSettings`:
`adamw_learning_rate=0.0003`, `muon_momentum=0.95`, `muon_nesterov=true`,
`muon_ns_steps=5`, `betas=[0.9,0.95]`, `weight_decay=0`, `warmup_steps=0`, `sequence_length=512`,
`packing_stride=sequence_length`, `seed=1337`, `checkpoint_interval_steps=1000`,
`checkpoint_keep_last=2`, `tokenization_batch_size=1000`, `use_bf16=false`,
`use_compile=true`, `compile_threads=4`, `deterministic=false`, `log_interval_steps=100`,
`shuffle_per_epoch=true`, `shuffle_data=false`, `gradient_clip_norm=1`, `max_validation_tokens=32768`,
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
counts. This keeps the model's input shape fixed through epoch tails. Set
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
checkpoint overhead. Epoch ends also report progress. The queue logs each
backpressure wait and its running total.

`shuffle_per_epoch` shuffles passages within each
source with `seed + epoch`, preserving token order inside each passage.
`shuffle_data` randomly interleaves packed windows across all active sources
before batching, while preserving each source's exact token allocation. Mixing
uses bounded memory and samples sources according to their remaining window
counts. With `shuffle_per_epoch=false`, the mixed order repeats each epoch;
with it enabled, both passage order and source mixing change each epoch.
Both controls are reproducible from `seed` and are shared by training and preview.

When `shuffle_data=false`, sources are traversed in the fixed order BabyLM,
FineWeb-Edu, Cosmopedia Khan Academy, and Cosmopedia stories. The legacy `shuffle`
setting remains an alias for `shuffle_per_epoch`; old runs retain unmixed source
order unless a fresh run enables `shuffle_data`.

Documents within each source form a continuous causal stream before packing;
BOS/EOS mark their boundaries without blocking
attention between documents. A stride below the context length repeats targets
across overlapping windows, and those repeats count toward the allocation.

Each source must supply its allocated targets in a single epoch. A
capacity check rejects larger allocations even when multiple epochs are
requested. FineWeb
reserves the first 0.05 GB of source text for validation even when its training
allocation is zero. BabyLM groups consecutive source text into 256-word passages,
preserving whitespace and adding document markers once per passage.

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

Preview shares training's tokenization, epoch plan, seed, stride, padding, and
batch iterator. Source labels travel with the packed windows. It skips held-out
validation downloads. `--first-source-only` avoids later sources when the first
source's first epoch supplies every requested window; otherwise omit the flag.

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
Resume also checks the token budget scope so saved learning-rate schedules
match the configured per-epoch token allocations.

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

Reusable inference functions live in `fraise.inference`. The notebooks in
`notebooks/` call the same run loader, scoring, generation, and benchmark
functions. Run loading preserves weight tying and the archived tokenizer.
Generation and the benchmark decode with a key/value cache and project only
the last position onto the vocabulary. Once a sequence exceeds `max_seq_len`, the
context window slides and each step recomputes the visible window. Positive
sampling temperatures work with the default nucleus sampler; optional top-k
filtering can also be supplied.

## Existing artifacts

Existing tokenizer JSON files remain loadable without migration if they contain
the required special tokens. The old tokenizer metadata index has an explicit
migration that preserves tokenizer bytes and saves `metadata.v2.json`:

```sh
uv run create-tokenizer --migrate-metadata
```

Version-2 checkpoints and old final model weights load through a small key-name
migration for the former embedding/output wrappers and SwiGLU spelling. To update
an old run's metadata, explicitly rebuild its manifest from its archived settings
and the current source cache:

```sh
uv run migrate-run data/models/OLD_RUN
```

This can download missing source prefixes. It records a new dataset identity,
keeps the original manifest and `model_meta.v2.json`, and leaves checkpoint/model
bytes untouched. Use the same source text as the original experiment; migration
cannot prove that re-downloaded unpinned remote sources match a historical cache.
Migration preserves the recorded AdamW optimizer identity and does not convert
optimizer state to Muon.
The old top-level Python import paths and redundant CLI aliases have been removed.

## Storage and development

Raw streaming downloads have request-specific directories under
`DATA_DIR/raw/SOURCE/raw/`. Validation uses its own request cache and survives
changes to training allocations. Arrow caches use local hard links when possible;
source changes invalidate them. Tokenized prefixes are memory-mapped int32 token
caches on disk, reused across epochs and resumes; the first run after upgrading
from int64 caches rebuilds them once. Packing reads Arrow buffers directly as NumPy
arrays, and a resumed epoch jumps to its next window without re-packing earlier
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

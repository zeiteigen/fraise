Fraise explores a language-model architecture with **decoupled input and output vocabularies**: a large input vocabulary for lexical compression and a smaller output vocabulary for tractable autoregressive generation.

Large **macro-tokens** reduce expensive global transformer timesteps and can encode richer lexical units, while smaller **base tokens** preserve token-internal structure—the project is named after the famous _strawberry_ problem.

Macro-tokens are composed from their base-token decomposition and can be **dynamically expanded** when finer structure is useful. This gives the model variable representational granularity and a mechanism for dynamically allocating transformer compute and KV-cache entries.

During generation, the smaller output vocabulary is bridged back into the macro vocabulary using a **token trie**. The model generates base tokens until they form a valid macro-token, which is then fed back into the global transformer. **Multi-Token Prediction (MTP)** can accelerate this local decoding step by predicting several base tokens at once.

The central question is:

> **Can aggressive lexical compression reduce global transformer compute while preserving access to fine-grained structure when needed?**

## Training configuration

Training reads two type-checked JSON files: one for training hyperparameters and
one for the model configuration. It also requires the tokenizer JSON file to
use for the run. Running `train` without these options prompts for each path.

```sh
uv run train \
  --training-hyperparameters configs/training_hyperparameters.json \
  --model-config configs/model_config.json \
  --tokenizer-file /path/to/fraise-8192-dataset-name.json
```

To inspect the first 100 examples in the exact training order, run:

```sh
uv run visualize-training-data \
  --training-hyperparameters data/h.json \
  --model-config data/mc.json \
  --tokenizer-file data/tokenizers/fraise-8192-babylm-cosmopedia-khanacademy-cosmopedia-stories-fineweb-edu.json \
  --first-source-only \
  --output data/training_examples.html
```

Open the HTML file to see decoded input and target windows, their source and
batch position, and the complete token IDs and target mask. The command uses
the same source loading, tokenization, epoch budgets, shuffle seed, stride,
padding, and batch iterator as `train`. The shown configuration's first 100
windows all come from BabyLM; `--first-source-only` avoids loading later sources
that cannot affect these windows. Omit that flag to load and validate every
configured training source. Use `--count` to change the number of examples.

If a path is not passed on the command line, enter it when prompted. The two
configuration files are copied into the model run directory.
The run directory also contains the exact tokenizer used for training as
`tokenizer.json`, so inference does not depend on the local tokenizer cache.

Set `"tie_word_embeddings": true` in the model configuration to share the
LM head's weight matrix with the token embeddings. The LM head keeps its own
bias. This option defaults to `false` when omitted, and `uv run params` accounts
for the shared weights.

Create a tokenizer with `uv run create-tokenizer`. It prompts for the
vocabulary size and writes the tokenizer under
`data/tokenizers/fraise-<vocab-size>-<dataset-name>.json`. Its metadata record
is stored in `data/tokenizers/metadata.json`. Pass the resulting file path to
`train` with `--tokenizer-file`.

Training hyperparameters include an absolute target-token count for every
configured training dataset. Counts must be non-negative integers and sum exactly
to `max_train_tokens`. Use `0` to exclude a source. Training checks
each source's tokenized capacity before the model starts; `num_epochs` allows
the same source text to be used again until its configured count is met. These
counts are causal target positions from the selected Fraise tokenizer, not
BabyLM's whitespace-separated word count.

`batch_size` is the number of packed sequences per optimizer step. To reduce
the memory used by each forward pass, set `minibatch_size` and/or
`gradient_accumulation_steps`. Their product must equal `batch_size`; either
one can be omitted and will be derived from the other. For example,
`"batch_size": 64, "minibatch_size": 8` runs eight minibatches before each
optimizer step. Gradients are weighted by the number of valid target tokens,
including in the final partial batch. Without these settings, training uses
one minibatch per optimizer step.

```json
{
  "max_train_tokens": 250000000,
  "dataset_token_counts": {
    "babylm": 100000000,
    "fineweb-edu": 150000000,
    "cosmopedia-khanacademy": 0,
    "cosmopedia-stories": 0
  }
}
```

When a dataset contributes less than its complete token stream, pretraining
uses its tokens from the beginning of the dataset rather than selecting random
examples.

For FineWeb-Edu and Cosmopedia, training stops the source download once the
selected tokenizer has supplied enough training token IDs for that source's
allocation, or when the source's byte limit is reached. The tokenizer
fingerprint and allocation are part of the download cache key. BabyLM
continues to use its pinned native text-file snapshot.

Downloaded source files live under `data/raw/`. Their Arrow shards are pinned
under each source's `.arrow-cache/` directory and reopened on later runs, so
the source files do not need to be parsed again. Local hard links keep the
Arrow cache from copying the large Hugging Face cache files on the same disk.
Changing a source file or download request creates a fresh Arrow cache. The
training command also skips loading and tokenizing sources assigned zero
training tokens; FineWeb validation remains available when its training count
is zero and downloads only the validation portion. Old `.arrow-cache/`
directories can be removed to reclaim space;
they are rebuilt when needed.

Training retains the two most recent resumable checkpoints by default. Set
`checkpoint_keep_last` in the training hyperparameters to retain a different
number; checkpoints still needed by pending validation are kept until that
validation finishes. Resuming from a run directory reads its model name from
the run metadata when `--model-name` is omitted.

The BabyLM loader groups consecutive text into 256-word rows, keeping the
original newlines. Tokenization adds BOS and EOS once per row, so these markers
are added once per passage rather than once per source line.

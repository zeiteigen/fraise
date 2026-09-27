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

If a path is not passed on the command line, enter it when prompted. The two
configuration files are copied into the model run directory.
The run directory also contains the exact tokenizer used for training as
`tokenizer.json`, so inference does not depend on the local tokenizer cache.

Create a tokenizer with `uv run create-tokenizer`. It prompts for the
vocabulary size and writes the tokenizer under
`data/tokenizers/fraise-<vocab-size>-<dataset-name>.json`. Its metadata record
is stored in `data/tokenizers/metadata.json`. Pass the resulting file path to
`train` with `--tokenizer-file`.

Training hyperparameters include an absolute target-token count for every
loaded training dataset. Counts must be non-negative integers and sum exactly
to `max_train_tokens`. Use `0` to exclude a loaded source. Training checks
each source's tokenized capacity before the model starts; `num_epochs` allows
the same source text to be used again until its configured count is met. These
counts are causal target positions from the selected Fraise tokenizer, not
BabyLM's whitespace-separated word count.

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

The BabyLM loader groups consecutive text into 200-word rows, keeping the
original newlines. Tokenization adds BOS and EOS once per row, so these markers
are added once per passage rather than once per source line.

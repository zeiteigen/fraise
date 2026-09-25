Fraise explores a language-model architecture with **decoupled input and output vocabularies**: a large input vocabulary for lexical compression and a smaller output vocabulary for tractable autoregressive generation.

Large **macro-tokens** reduce expensive global transformer timesteps and can encode richer lexical units, while smaller **base tokens** preserve token-internal structure—the project is named after the famous _strawberry_ problem.

Macro-tokens are composed from their base-token decomposition and can be **dynamically expanded** when finer structure is useful. This gives the model variable representational granularity and a mechanism for dynamically allocating transformer compute and KV-cache entries.

During generation, the smaller output vocabulary is bridged back into the macro vocabulary using a **token trie**. The model generates base tokens until they form a valid macro-token, which is then fed back into the global transformer. **Multi-Token Prediction (MTP)** can accelerate this local decoding step by predicting several base tokens at once.

The central question is:

> **Can aggressive lexical compression reduce global transformer compute while preserving access to fine-grained structure when needed?**

## Training configuration

Training reads two type-checked JSON files: one for training hyperparameters and
one for the model configuration. Running `train` without the corresponding
options prompts for both paths. Example files matching the previous defaults
are available in `configs/`.

```sh
uv run train \
  --training-hyperparameters configs/training_hyperparameters.json \
  --model-config configs/model_config.json
```

If the files are not passed on the command line, enter their paths when
prompted. Both source files are copied into the model run directory.

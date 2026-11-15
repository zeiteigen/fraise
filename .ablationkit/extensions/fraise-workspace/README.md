# fraise-workspace

fraise-web's **Autocomplete**, **Compare** and **Train** views as AblationKit project panels. With
the [tokenization workbench](../tokenization-workbench/README.md) they replace the old
`fraise-web` server.

| Panel | What it does |
| --- | --- |
| **Autocomplete** | Edit a document and continue the text before the cursor with a saved checkpoint. The continuation streams in; Accept (Tab) inserts it at the saved cursor (replacing a selection), Discard (Esc) drops it, Ctrl/Cmd+Enter generates. The model gets raw text, without roles or chat templates. Inputs, checkpoint and sampling settings are saved in `DATA_DIR/workspace/saved-inputs.json` (newest 40) and can be restored. The inspector shows the run's dimensions, vocabulary, context limit, checkpoint step and validation. |
| **Compare checkpoints** | Send one prompt with the same sampling settings and seed to up to eight checkpoints, one after another, with speed, load time, prompt tokens and each run's saved tokenizer, configuration and validation. |
| **Train** | Configure a fresh run (architecture preset or a saved run's configuration; form fields or full JSON; dataset token budgets) or resume a saved training checkpoint (only operational settings change), check it with fraise's validation and parameter count, and **Prepare** it: a plan in `DATA_DIR/launch/<run>/<time>/`. |

## Training runs as an AblationKit job

Training is not started from the panel. The prepared plan is run by the repository's `MLproject`
entrypoint `launch` (Runs → More → Entrypoints → `launch` → Run…, with the plan directory the
panel shows), so it is an ordinary AblationKit job:

- **Progress:** the job page has the console; the run page has `loss`, `gradient_norm`,
  `tokens_per_second`, `tokens_seen`, `total_tokens` and `validation/loss`,
  `validation/bits_per_byte`, by global step (`train --ablationkit-metrics`, the same names an
  agent job's metrics-file collection gives).
- **Stop:** Cancel the job. AblationKit sends SIGTERM; fraise saves a resumable checkpoint after
  the current optimizer step and exits (AblationKit allows 15 s before SIGKILL). A job's time
  limit stops it the same way. Resume it from the Train panel's **Resume run**.

The `train` entrypoint runs a fresh run from configuration files without the panel.

## How it works

- **Python** (`python/workspace.py`) runs in fraise's own environment at the checkout's current
  commit, in a warm worker, so the loaded checkpoint and tokenizer fingerprints stay between
  runs. It calls `fraise.workbench`: `Catalog` (runs in `DATA_DIR/models`, tokenizers in
  `DATA_DIR/tokenizers`), `generate_sequentially` with an `InferenceSession`,
  `validate_training`/`write_plan` and `SavedInputs`. It reimplements nothing.
- **Streaming:** while a checkpoint generates, the execution's progress message carries the
  output's length and its last ~480 characters (AblationKit keeps 500), polled about twice a
  second. The view stitches them together; if more than that is generated between two polls it
  shows `…` until the complete outputs arrive with the result.
- **Device:** CUDA when available (`python.gpu: true`). The last checkpoint stays loaded until
  another one is needed, the worker idles out (10 minutes), or **Release device memory** is used.
  Nothing coordinates the panels with training jobs (fraise-web's server unloaded the model and
  refused generation while it trained), so release the model before training on the same GPU.
- **Limits:** a generation can run for 15 minutes; the worker may use 12 GiB of memory. There is
  no Stop for a generation in progress (the bridge has no cancel); lower *Max new tokens* for
  quick drafts.
- **Capabilities:** `project.analysis.execute` (its own Python) and `gpu`. Python has no network;
  it reads fraise's data directory and writes saved inputs and launch plans there, as trusted
  code under your user.

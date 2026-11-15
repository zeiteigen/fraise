<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261111-02a6bd","project_id":"fraise","kind":"note","tags":["naming","housekeeping"],"links":[{"kind":"study","id":"vocab-gap-audit-38m"},{"kind":"study","id":"poc-pilot-100m"},{"kind":"study","id":"lr-sweep-38m"},{"kind":"task","id":"task-20261111-ljvjsb"}],"title":"Renamed tokenizers, saved models and configs: old → new names","created_at":"2026-11-11T07:58:34.768Z","updated_at":"2026-11-11T07:58:34.768Z","created_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"updated_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"revision":1} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Renamed tokenizers, saved models and configs: old → new names

_ID `note-20261111-02a6bd` · tags naming, housekeeping · agent, 2026-11-11T07:58:34.768Z_

Links:

- study `vocab-gap-audit-38m`
- study `poc-pilot-100m`
- study `lr-sweep-38m`
- task `task-20261111-ljvjsb`

<!-- ablationkit:body -->
On 2026-11-11 the local files were renamed. Models are now named by the tokenizer they were trained on, not "base"/"symmetric": both the 8k and 64k transformers are symmetric, and every earlier "symmetric 64k" model used the macro tokenizer. The files and weights are unchanged. **MLflow run names, recorded job commands and earlier records still use the old names.** Use this table, or `data/RENAMES.json`, to translate them.

**Tokenizers** (`data/tokenizers/`; symlinks in `data/experiments/poc-fineweb/tokenizers/`)

| Old | New | What it is |
|---|---|---|
| `base-8192.json` | `bpe-8k.json` | 8k byte-level BPE used by every 8k/8k run (sha 1fbe2820) |
| `fraise-macro-65536-b501b7e0c0e323b6.json` | `macro-64k.json` | 64k macro tokenizer over `bpe-8k`, merges across words. Every 64k/64k run before the gap audit used it (pilot, sweep, gap-macro64k-*). |
| `fraise-65536-456cef19bebbe6d6.json` | `bpe-64k.json` | standard 64k byte-level BPE on the same FineWeb-Edu slice; used only by gap-bpe64k-emb002-finalnorm |
| `fraise-65536-babylm-fineweb-edu.json` | `bpe-64k-legacy.json` | older standard 64k BPE (BabyLM + FineWeb-Edu); no recorded run used it. It shares 55,855 of 65,536 tokens with `bpe-64k`. |
| `fraise-8192.json` | `bpe-8k-legacy.json` | older 8k BPE (BabyLM + FineWeb-Edu), not used by the pilot, sweep or audit |
| `fraise-131072-babylm-fineweb-edu.json` | `bpe-128k-legacy.json` | older 128k BPE, unused |
| `fraise-8192-babylm-cosmopedia-khanacademy-cosmopedia-stories-fineweb-edu.json` | removed | byte-identical copy of `bpe-8k.json` |

**Saved models** (folder, weights file and `model_name`)

| Old | New |
|---|---|
| pilot-base / pilot-symmetric / pilot-fraise | pilot-bpe8k / pilot-macro64k / pilot-fraise |
| sweep-base-LR / sweep-symmetric-LR / sweep-fraise-LR | sweep-bpe8k-LR / sweep-macro64k-LR / sweep-fraise-LR |
| gap-base-{default, emb002, emb002-finalnorm} | gap-bpe8k-{…} |
| gap-sym-{default, emb002, emb002-finalnorm, emb002-finalnorm-b32} | gap-macro64k-{…} |
| gap-bpe64k-emb002-finalnorm | unchanged |

**Configs:**
- `data/ak/configs/gap`: `base-lr1e-3*` → `bpe8k-lr1e-3*`, `sym-lr3e-3*` → `macro64k-lr3e-3*`, `model-base*` → `model-8k*`, `model-symmetric*` → `model-64k*`. The 64k model config is shared by the macro and standard-BPE runs.
- `data/experiments/poc-fineweb/configs`: `base.json` → `8k.json` and `symmetric.json` → `64k.json` (also in `100m/`), `pilot-base.json` → `pilot-bpe8k.json`, `pilot-macro.json` → `pilot-macro64k.json`.

The experiment scripts, analysis scripts, logs and the tokenization workbench defaults now use the new names. The rename script is `data/ak/analysis/rename_artifacts.py`, and its log is next to it.

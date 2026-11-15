<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261110-4db192","project_id":"fraise","kind":"note","tags":["onboarding"],"links":[{"kind":"study","id":"poc-pilot-100m"},{"kind":"study","id":"lr-sweep-38m"},{"kind":"run","id":"26892877924b49d7887659b86cb676a6"},{"kind":"job","id":"j-ebbc6375db9d"},{"kind":"task","id":"task-20261110-apoxgb"}],"title":"Onboarding","created_at":"2026-11-10T20:52:33.531Z","updated_at":"2026-11-11T07:34:56.443Z","created_by":{"kind":"agent","id":"6830bb2a421e119b"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":2,"summary":"How fraise was connected to AblationKit on 2026-11-10. Training runs as tracked jobs through the project's own train command, with its events file as the metrics source, and a 200-step smoke run proved that metrics, including validation bits per byte (BPB), reach the run page. The 12 earlier runs of the 100M pilot and the 38M learning-rate sweep were imported from their files (not re-run), and 11 older or scratch runs were left out by the researcher's choice. It also lists where each earlier visualisation went and the known gaps: job parameters are not logged as MLflow params, imported runs have no AblationKit provenance, and every configuration has a single seed."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Onboarding

_ID `note-20261110-4db192` · tags onboarding · agent, 2026-11-10T20:52:33.531Z_

Links:

- study `poc-pilot-100m`
- study `lr-sweep-38m`
- run `26892877924b49d7887659b86cb676a6`
- job `j-ebbc6375db9d`
- task `task-20261110-apoxgb`

How fraise was connected to AblationKit on 2026-11-10. Training runs as tracked jobs through the project's own train command, with its events file as the metrics source, and a 200-step smoke run proved that metrics, including validation bits per byte (BPB), reach the run page. The 12 earlier runs of the 100M pilot and the 38M learning-rate sweep were imported from their files (not re-run), and 11 older or scratch runs were left out by the researcher's choice. It also lists where each earlier visualisation went and the known gaps: job parameters are not logged as MLflow params, imported runs have no AblationKit provenance, and every configuration has a single seed.

<!-- ablationkit:body -->
Onboarded on 2026-11-10 by a Claude Code agent session, with the researcher's approval of the plan.

## How runs are tracked
- **Environment:** uv (`pyproject.toml` + `uv.lock`; the lock is current). Jobs run `uv run --frozen train …` from the checkout root.
- **Entrypoint:** `train` (`src/fraise/cli/train.py`). All configuration comes from JSON files (`--training-hyperparameters`, `--model-config`, `--tokenizer-file`); there are no step-limit flags, so short runs need their own hyperparameter file.
- **Metrics (path A, metrics file):** pass `--events-file data/ak/events/<run>.jsonl`, and give the job `metrics_file` = the same path with `metrics_format {"step_keys": ["global_step"], "timestamp_key": "time"}`.
  - Keys: `loss` (train, per target: per macro for 64k models), `gradient_norm`, `tokens_seen` (base tokens), `tokens_per_second`, `epoch`, `total_tokens`, `validation/loss`, `validation/bits_per_byte`.
  - Commit 72b1efb added a flat `validation_score` event in `src/fraise/training/run/lifecycle.py`. Before it, validation results reached the events file only as arrays, which metric readers ignore. Training is unchanged; all 221 unit tests passed.
- **Params:** not logged as MLflow params for jobs (no SDK calls). The configs are recorded as `inputs/…` artifacts (`input_paths`) and are visible in the command.
- **Artifacts:** the run directory's `model_meta.json`, `model_config.json`, `training_hyperparameters.json`, `tokenizer.json` and `training.log`, plus the events file. Checkpoints and final weights (`*.pt`, ~1.3 GB per checkpoint at 100M) are excluded.
- **Data:** `--data-dir data/experiments/poc-fineweb` reuses the experiment's tokenizers and the shared raw FineWeb-Edu cache (`raw -> ../../raw`). `--save-path data/ak/runs` keeps AblationKit job runs apart from the historical ones. All of `data/` is git-ignored.

## Smoke run
- Job `j-ebbc6375db9d`, MLflow run `26892877924b49d7887659b86cb676a6` (`smoke-base-01`): base 8k at 100M dense, pilot-base settings except a 6,553,600-token budget (200 steps), warmup 20, a checkpoint every 50 steps, and 32,768 validation tokens.
- Outcome: succeeded, synced, commit 72b1efb (clean), limits enforced (systemd). Validation BPB at steps 50, 100, 150 and 200 matches `training.log`.
- It is a tracking check only, not a result, and is in neither study.
- Throughput was ~45–54k base tokens/s, against ~81k in the pilot log for the same model. Not investigated: short logging intervals and a cold GPU are possible causes.

## Earlier runs
- **Imported (12, labelled as imports):** `pilot-{base,symmetric,fraise}` → study `poc-pilot-100m`; `sweep-{base,symmetric,fraise}-{0.003,0.001,0.0003}` → study `lr-sweep-38m`.
  - Source: `data/experiments/poc-fineweb/models/<run>`. `data/models/<run>` holds the same files.
  - Histories were written by `data/ak-import/convert.py` (attached to each import), which copies the `Step N:` lines of `training.log` and the full-precision `validation_results` of `model_meta.json` under the same keys as new runs. The import cross-checks every logged validation value against the metadata.
  - Each import records commit d0fc54e and tag `source_git_dirty`. These runs ran from the worktree `~/.t3/worktrees/fraise/t3-0e6f556d`, not from this checkout.
- **Not imported, by the researcher's choice (11):**
  - Older code (schema 2, BabyLM-era tokenizers, validation loss only, no BPB): `data/models/minsorhours` and `data/models/fraise_model` (finished, 7,630 steps, commit ab3b941); `data/models/unwantedchild` (stopped at 6,766 steps, 221.7M of 2.5B tokens, commit c24cff1); `data/models/bbb` (24,000 steps, no end time or final model, commit fc28506).
  - Scratch and test runs (commit 16c0e1c, dirty): `overfit`, `overfitfine`, `overfitthousand`, `overfittwo`, `stdtiedweights`, `testingthiscommit`, `whyis`, all under `data/models/`.
  - The diagnostics runs `data/diagnostics/muon-smoke-20261005/runs/baseline` and `data/diagnostics/document_training/crops` were also left out.

## Visualisations
- Report §5.1 (sweep table) and §5.2 (BPB by step): native charts. Open the studies, or Runs → Compare on `validation/bits_per_byte`.
- Web UI live loss chart: the native run chart for new jobs.
- §5.3 benchmark (`bench.log`) and §5.5 post-hoc analyses (`analyze*.py`, `chain.py`, `analysis*.log`, `chain.log`): attached as artifacts to the `pilot-fraise` import.
- §5.4 compute-axis comparison: extension `compute-axes` was scaffolded but **not written or built**. The agent's harness denies `.ablationkit/`, so it could not edit the template; see the follow-up in the onboarding report.
- Not migrated, because they are not run outputs: `docs/article/figures/*` (concept diagrams), `data/training_examples.html` (data preview), the web Tokenization workbench (stays in `fraise-web`).

## Gaps
- Job params are not queryable as MLflow params.
- Imported runs did not run under AblationKit; their commits come from their own metadata, and the pilots were dirty.
- One seed per configuration everywhere, so seed variance is unknown.
- The imported histories lack `epoch` and `total_tokens`, which new runs log.
- No datasets or models are registered (no SDK calls).

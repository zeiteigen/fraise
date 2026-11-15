<!-- ablationkit:record {"schema_version":1,"record_id":"rep-20261110-94dd95","project_id":"fraise","kind":"report","tags":["onboarding"],"links":[{"kind":"study","id":"poc-pilot-100m"},{"kind":"study","id":"lr-sweep-38m"},{"kind":"note","id":"note-20261110-4db192"},{"kind":"job","id":"j-ebbc6375db9d"},{"kind":"task","id":"task-20261110-apoxgb"}],"status":"final","title":"Onboarding report: fraise","summary":"Onboarding fraise: tracking works end to end, shown by a 200-step smoke run of the 8k/8k transformer whose training and validation bits-per-byte (BPB) metrics synced, and the research context is recorded: two studies, 12 imported earlier runs, three hypotheses, three approaches, one decision and three notes. One criterion was not met at the time: the compute-axis comparison view was scaffolded but not written or built, because the agent could not edit the extension's files; it was built afterwards (see the follow-up note). The smoke run checks tracking only and is not a research result.","follow_up":"1. Let the agent harness edit .ablationkit/extensions/compute-axes (or write the view yourself), then build it and approve it in Project settings → Extensions. *(Since done up to approval; see the follow-up note.)*\n2. Run a tracked full baseline (pilot-base config, ~1.8 h, limit ~3 h = 3 of 10 weighted hours) so the study has one run made under AblationKit. Seeds need more budget than the 10 h limit leaves.\n3. Optionally investigate the smoke run's lower throughput (~45-54k vs ~81k base tokens/s in the pilot).","evidence":[{"ref":{"kind":"run","id":"26892877924b49d7887659b86cb676a6"},"role":"candidate","mlflow_run_id":"26892877924b49d7887659b86cb676a6","values":{"validation/bits_per_byte":"1.775600375299282"},"note":"Smoke run of the tracked baseline invocation (base 8k, 200 steps). Shows that tracking works; it is not a research result.","captured_at":"2026-11-10T20:53:34.318Z"}],"created_at":"2026-11-10T20:53:34.319Z","updated_at":"2026-11-11T07:38:11.200Z","created_by":{"kind":"agent","id":"6830bb2a421e119b"},"updated_by":{"kind":"agent","id":"6ef314f6613932d5"},"revision":2} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Onboarding report: fraise

_ID `rep-20261110-94dd95` · status final · tags onboarding · agent, 2026-11-10T20:53:34.319Z_

Links:

- study `poc-pilot-100m`
- study `lr-sweep-38m`
- note `note-20261110-4db192`
- job `j-ebbc6375db9d`
- task `task-20261110-apoxgb`

## Evidence summary

Onboarding fraise: tracking works end to end, shown by a 200-step smoke run of the 8k/8k transformer whose training and validation bits-per-byte (BPB) metrics synced, and the research context is recorded: two studies, 12 imported earlier runs, three hypotheses, three approaches, one decision and three notes. One criterion was not met at the time: the compute-axis comparison view was scaffolded but not written or built, because the agent could not edit the extension's files; it was built afterwards (see the follow-up note). The smoke run checks tracking only and is not a research result.

## Evidence

| Role | Variant | Object | Attempt | Source | Values |
| --- | --- | --- | --- | --- | --- |
| candidate |  | run `26892877924b49d7887659b86cb676a6` |  |  | validation/bits_per_byte=1.775600375299282 |

## Follow-up

1. Let the agent harness edit .ablationkit/extensions/compute-axes (or write the view yourself), then build it and approve it in Project settings → Extensions. *(Since done up to approval; see the follow-up note.)*
2. Run a tracked full baseline (pilot-base config, ~1.8 h, limit ~3 h = 3 of 10 weighted hours) so the study has one run made under AblationKit. Seeds need more budget than the 10 h limit leaves.
3. Optionally investigate the smoke run's lower throughput (~45-54k vs ~81k base tokens/s in the pilot).

## Interpretation

<!-- ablationkit:body -->
The smoke run proves that tracking works. It is not evidence for or against any research claim. The study reports `rep-20261110-90119f` and `rep-20261110-7d1283` restate imported historical results; they are not new measurements.

## Criteria
1. **Smoke run synced with its metrics, including validation/bits_per_byte: met.** Job `j-ebbc6375db9d`, run `26892877924b49d7887659b86cb676a6`.
   - State: succeeded, synced, commit 72b1efb (clean), limits enforced (systemd), `inputs/` artifacts present for both configs.
   - `validation/bits_per_byte` was logged at steps 50, 100, 150 and 200 (final 1.775600375299282, a 200-step smoke model), matching `training.log`.
2. **Studies `poc-pilot-100m` and `lr-sweep-38m` recorded: met.**
3. **12 earlier runs imported with their histories: met.** No import warnings. One history per kind of source was spot-checked (`pilot-fraise` validation BPB against pilot report §5.2), and the final values match the report's §5.1 and §5.2 tables.
4. **Visualisations migrated or listed; §5.4 extension built and awaiting approval: NOT met.**
   - The other figures are migrated or listed (onboarding note `note-20261110-4db192`).
   - Extension `compute-axes` was scaffolded from `result-comparison`, but the agent harness denies reading or editing `.ablationkit/`, so its code is still the unmodified template. It was deliberately not built.
5. **Hypotheses, decision and notes recorded as approved: met.**
   - Hypotheses: `hyp-20261110-76aef8`, `hyp-20261110-67fe7f`, `hyp-20261110-fd912f`.
   - Decision: `dec-20261110-d16d1e` (accepted).
   - Approaches: `apr-20261110-92615e`, `apr-20261110-1d5459`, `apr-20261110-bd6f12`.
   - Notes: `note-20261110-4db192`, `note-20261110-a8bb9b`, `note-20261110-e157db`.

## Since this report
- The compute-axis comparison view was written and built as a project extension, awaiting approval at the time (follow-up note "§5.4 compute-axis extension built").
- Each view of the repository's own web app was given a destination; the tokenization view was rebuilt as a project panel (follow-up note "fraise-web views migrated").
- The gap audit (November 2026) found two training faults in the imported transformer baselines, which qualifies the two imported studies (note "Pilot and LR-sweep baselines were trained with untrained embeddings and no final norm").

<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261110-716d44","project_id":"fraise","kind":"note","tags":["onboarding","extension"],"links":[{"kind":"report","id":"rep-20261110-94dd95"},{"kind":"study","id":"poc-pilot-100m"},{"kind":"note","id":"note-20261110-4db192"}],"title":"§5.4 compute-axis extension built (compute-axes), awaiting approval","created_at":"2026-11-10T21:05:52.183Z","updated_at":"2026-11-11T07:34:56.456Z","created_by":{"kind":"agent","id":"2d28ed65c4b3562e"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":2,"summary":"Follow-up to onboarding: the compute-axis comparison view, which the onboarding report listed as not built, was written and built as a project extension and, at the time of writing, awaited the researcher's approval. In Runs, Compare, it reads a reference run's validation bits-per-byte (BPB) curve at the same point as each candidate on one of four axes: equal tokens, equal training positions, equal wall-clock, or equal FLOPs from a figure the researcher types in. Run on the imported pilot histories it reproduces the pilot report's readings, for example 1.22489 BPB for the 8k/8k transformer at equal positions against the report's 'at most about 1.225'; these are re-readings of imported runs, not new measurements."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# §5.4 compute-axis extension built (compute-axes), awaiting approval

_ID `note-20261110-716d44` · tags onboarding, extension · agent, 2026-11-10T21:05:52.183Z_

Links:

- report `rep-20261110-94dd95`
- study `poc-pilot-100m`
- note `note-20261110-4db192`

Follow-up to onboarding: the compute-axis comparison view, which the onboarding report listed as not built, was written and built as a project extension and, at the time of writing, awaited the researcher's approval. In Runs, Compare, it reads a reference run's validation bits-per-byte (BPB) curve at the same point as each candidate on one of four axes: equal tokens, equal training positions, equal wall-clock, or equal FLOPs from a figure the researcher types in. Run on the imported pilot histories it reproduces the pilot report's readings, for example 1.22489 BPB for the 8k/8k transformer at equal positions against the report's 'at most about 1.225'; these are re-readings of imported runs, not new measurements.

<!-- ablationkit:body -->
Follow-up to the onboarding report `rep-20261110-94dd95`, criterion 4. The onboarding task stays closed.

## What was built
Extension `compute-axes` v0.1.0, build `eb-fb50c1adb0e35b7f` (status `needs_approval`, no diagnostics). It is a `comparison.view` (Runs → Compare → "Compute-axis comparison"): `ui/src/Compare.tsx`, `ui/src/Chart.tsx`, `ui/src/axes.ts`, `README.md`.

For each selected candidate run it reads the final validation point. It then reads the chosen reference run's `validation/bits_per_byte` curve at the same x, on one of four axes:
- **Equal tokens:** `tokens_seen` at the validation step.
- **Equal training global positions:** step × `batch_size` × `sequence_length` (params).
- **Equal wall-clock:** time of the training point at that step minus the time of the first logged training point, so it excludes data preparation, compilation and the first log interval.
- **Equal FLOPs:** tokens × a FLOPs-per-base-token figure the researcher types in, since no run records FLOPs.

Reference values are interpolated linearly between validation points and never extrapolated. The view states the report's caveats: the reference is read before its LR annealed, and there is one seed per configuration. It also shows a per-run table of every axis.

**Capability:** `project.runs.read` only (enforced). It calls `runs.get` for params and `runs.metric_history` for the BPB and `tokens_seen` histories. It writes nothing and has no Python.

## Checks
- Offline validation passed; a `tsc` strict type-check against the SDK sources passed.
- A scratch script ran `axes.ts` on the imported pilot histories (`data/ak-import/pilot-*.jsonl`) with base 8k as the reference. It reproduces report §5.4:
  - equal tokens: base 1.130199561195575 at 500M base tokens (symmetric +0.064, Fraise +0.103);
  - equal positions: 293,208,064 positions, base read as 1.22489 (interpolated between steps 8,000 and 10,000; report "≤ ~1.225");
  - equal wall-clock: base read as 1.14279 for symmetric and 1.13098 for Fraise (report "≤ ~1.143 / ≤ ~1.131").
- Not yet seen rendered: the review preview refuses bridge calls, so the chart's layout needs a look after approval.

These are re-readings of imported runs, not new measurements.

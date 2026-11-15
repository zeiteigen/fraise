<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261110-a8bb9b","project_id":"fraise","kind":"note","tags":["untracked-result"],"links":[{"kind":"study","id":"poc-pilot-100m"},{"kind":"hypothesis","id":"hyp-20261110-76aef8"},{"kind":"task","id":"task-20261110-apoxgb"}],"title":"Pilot report and résumé numbers with no imported run behind them (untracked)","created_at":"2026-11-10T20:52:42.446Z","updated_at":"2026-11-11T07:34:56.448Z","created_by":{"kind":"agent","id":"6830bb2a421e119b"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":3,"summary":"An inventory of numbers that appear in the repository's documents (the October pilot report and a résumé content bank) but have no tracked or imported run metric behind them, so a reader knows which figures AblationKit cannot check. They are the analytic training-FLOPs estimates per base token (about 666M for the 8k/8k transformer, 546M for the 64k/64k on the macro tokenizer and 520M for Fraise) and the compute-axis comparisons built on them, the pilot's steady-state throughput and peak-memory readings, and two older BabyLM-era runs cited in the résumé. The full-held-out bits-per-byte values with bootstrap intervals are not in the list, because they are in an analysis log attached to the imported Fraise pilot run."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Pilot report and résumé numbers with no imported run behind them (untracked)

_ID `note-20261110-a8bb9b` · tags untracked-result · agent, 2026-11-10T20:52:42.446Z_

Links:

- study `poc-pilot-100m`
- hypothesis `hyp-20261110-76aef8`
- task `task-20261110-apoxgb`

An inventory of numbers that appear in the repository's documents (the October pilot report and a résumé content bank) but have no tracked or imported run metric behind them, so a reader knows which figures AblationKit cannot check. They are the analytic training-FLOPs estimates per base token (about 666M for the 8k/8k transformer, 546M for the 64k/64k on the macro tokenizer and 520M for Fraise) and the compute-axis comparisons built on them, the pilot's steady-state throughput and peak-memory readings, and two older BabyLM-era runs cited in the résumé. The full-held-out bits-per-byte values with bootstrap intervals are not in the list, because they are in an analysis log attached to the imported Fraise pilot run.

<!-- ablationkit:body -->
Not re-run under AblationKit. These numbers appear in repository documents, but no imported run's metrics hold them.

**1. Analytic compute estimates.** Source: `docs/reports/2026-10-31-fraise-pilot-report.md` §5.4 (commit d0fc54e code state). They are derived estimates (6 × parameters per position plus ~33 MFLOP attention, divided by 1.705 positions per base token), not measurements:

| Model | Train FLOPs per base token | Global positions per base token |
| --- | ---: | ---: |
| Base 8k | ~666M | 1.00 |
| Symmetric 64k | ~546M | 0.59 |
| Fraise | ~520M | 0.59 |

The equal-FLOPs, equal-wall-clock and equal-positions comparisons in §5.4 read base BPB from the single base run's validation curve, interpolated between checkpoints before its LR had annealed. The report itself marks the equal-positions comparison "undetermined".

**2. Steady-state throughput and peak memory.** Source: pilot report §5.3 (base ~81k, symmetric ~94k, Fraise ~83k base tokens/s; peak 9.2 / 13.4 / 16.2 GB). The throughput readings are in the imported pilot runs (`tokens_per_second`), and peak memory is in their attached `training.log`, but the summary values are the report's readings, not a recorded metric. The minibatch benchmark is `bench.log`, attached to the `pilot-fraise` import.

**3. Résumé content bank.** Source: `data/fraise-project-resume.md` (2026-10-15). B02 (75.61M parameters, 250M target tokens, 7,630 steps, training CE 3.692468, validation CE 3.637749) cites `data/models/minsorhours/model_meta.json` (its source S4). B03 adds a 4.08M-parameter run on the same budget, which it cites as `data/models/fraise_model` (S5). Both run directories exist, but the researcher chose not to import them. The parameter counts are derived from configuration.

The full-reserve BPB values and bootstrap CIs in the report's summary table (1.1356 / 1.1989 / 1.2380; Fraise − symmetric paired CI [0.0386, 0.0395]) are **not** in this list: they are in `analysis2.log`, attached to the imported `pilot-fraise` run, and were computed post hoc by `analyze2.py` outside AblationKit.

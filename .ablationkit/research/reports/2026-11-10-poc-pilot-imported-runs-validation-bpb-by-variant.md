<!-- ablationkit:record {"schema_version":1,"record_id":"rep-20261110-90119f","project_id":"fraise","kind":"report","tags":["generated"],"links":[{"kind":"study","id":"poc-pilot-100m"},{"kind":"hypothesis","id":"hyp-20261110-76aef8"},{"kind":"hypothesis","id":"hyp-20261110-67fe7f"},{"kind":"task","id":"task-20261110-apoxgb"}],"status":"draft","title":"PoC pilot (imported runs): validation BPB by variant","summary":"Imported results of the 100M pilot: after 500M base tokens of FineWeb-Edu text, the 8k/8k transformer reached 1.1302 bits per byte (BPB), the 64k/64k transformer on the macro tokenizer 1.1940 (+0.064) and Fraise (64k in, 8k out) 1.2332 (+0.103), each from a single run imported from its files rather than re-run. At equal text the small-vocabulary baseline is clearly best, but the comparison is confounded: both transformers were later found to carry two training faults that penalise a 64k vocabulary, and the macro models took 17% more optimizer steps at a smaller batch and a higher learning rate.","evidence":[{"ref":{"kind":"run","id":"292b832a133a4155a076c99a033f9d06","label":"pilot-base"},"role":"baseline","mlflow_run_id":"292b832a133a4155a076c99a033f9d06","variant_id":"base-8k","values":{"validation/bits_per_byte":"1.130199561195575"},"note":"imported from data/experiments/poc-fineweb/models/pilot-base, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:15.896Z"},{"ref":{"kind":"run","id":"4340e898bf42422d85847701152c69f8","label":"pilot-symmetric"},"role":"candidate","mlflow_run_id":"4340e898bf42422d85847701152c69f8","variant_id":"symmetric-64k","values":{"validation/bits_per_byte":"1.19398661607335"},"note":"imported from data/experiments/poc-fineweb/models/pilot-symmetric, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:15.896Z"},{"ref":{"kind":"run","id":"fc8d59de94e549eabf30c7f29c9b5632","label":"pilot-fraise"},"role":"candidate","mlflow_run_id":"fc8d59de94e549eabf30c7f29c9b5632","variant_id":"fraise","values":{"validation/bits_per_byte":"1.233208708504531"},"note":"imported from data/experiments/poc-fineweb/models/pilot-fraise, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:15.896Z"}],"generated":{"study_id":"poc-pilot-100m","baseline_variant":"base-8k","metric_keys":["validation/bits_per_byte"],"runs":3,"generated_at":"2026-11-10T20:53:15.897Z"},"created_at":"2026-11-10T20:53:15.897Z","updated_at":"2026-11-11T07:38:11.204Z","created_by":{"kind":"agent","id":"6830bb2a421e119b"},"updated_by":{"kind":"agent","id":"6ef314f6613932d5"},"revision":3} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# PoC pilot (imported runs): validation BPB by variant

_ID `rep-20261110-90119f` · status draft · tags generated · agent, 2026-11-10T20:53:15.897Z_

Links:

- study `poc-pilot-100m`
- hypothesis `hyp-20261110-76aef8`
- hypothesis `hyp-20261110-67fe7f`
- task `task-20261110-apoxgb`

## Evidence summary

Imported results of the 100M pilot: after 500M base tokens of FineWeb-Edu text, the 8k/8k transformer reached 1.1302 bits per byte (BPB), the 64k/64k transformer on the macro tokenizer 1.1940 (+0.064) and Fraise (64k in, 8k out) 1.2332 (+0.103), each from a single run imported from its files rather than re-run. At equal text the small-vocabulary baseline is clearly best, but the comparison is confounded: both transformers were later found to carry two training faults that penalise a 64k vocabulary, and the macro models took 17% more optimizer steps at a smaller batch and a higher learning rate.

## Evidence

| Role | Variant | Object | Attempt | Source | Values |
| --- | --- | --- | --- | --- | --- |
| baseline | base-8k | run `292b832a133a4155a076c99a033f9d06` |  |  | validation/bits_per_byte=1.130199561195575 |
| candidate | symmetric-64k | run `4340e898bf42422d85847701152c69f8` |  |  | validation/bits_per_byte=1.19398661607335 |
| candidate | fraise | run `fc8d59de94e549eabf30c7f29c9b5632` |  |  | validation/bits_per_byte=1.233208708504531 |

## Interpretation

<!-- ablationkit:body -->
## Setup
Generated from the 3 runs of the study "PoC pilot at 100M": the same 99.2M-parameter transformer (d=768, 14 layers, context 512) in each model, 500,000,000 FineWeb-Edu base tokens per run, one seed (1337). The 8k/8k transformer used batch 64 × 512 and AdamW 1e-3 (15,259 steps); the two macro models used batch 32 × 512 and AdamW 3e-3 (17,896 steps). Validation every 2,000 steps over 262,144 held-out base tokens. All three runs were imported from files written before the project used AblationKit (commit d0fc54e, a separate dirty worktree); they were not run under it.

Bits per byte (BPB) is the summed negative log-likelihood of the held-out targets, in bits, divided by the UTF-8 bytes they cover, so it compares models with different tokenizers (lower is better):

$$
\mathrm{BPB} = \frac{1}{B \ln 2} \sum_{t} -\ln p(x_t \mid x_{<t})
$$

## Results
The generated table (latest logged value per run; deltas against the baseline are descriptive, no significance test):

| Variant | n | validation/bits_per_byte (↓) |
| --- | --- | --- |
| base-8k (baseline) | 1 | 1.1302 |
| fraise | 1 | 1.2332 (Δ +0.10301 worse) |
| symmetric-64k | 1 | 1.1940 (Δ +0.063787 worse) |

Parameters that differ between variants: `adamw_learning_rate`, `architecture`, `base_vocab_size`, `batch_size`, `local_dim`, `local_num_layers`, `local_swiglu_up_dim`, `macro_embedding`, `macro_residual_dim`, `max_macro_length`, `tie_word_embeddings`, `tokenizer_sha256`, `vocab_size`.

## Interpretation
- At equal text, the 8k/8k transformer is best, the 64k/64k model 0.064 BPB behind and Fraise 0.103 behind.
- Fraise against the 64k/64k model, which share tokenizer, data, batch, steps and transformer: Fraise led early (1.535 against 1.674 at about 56M tokens), the 64k/64k model overtook at about 208M tokens and finished 0.039 ahead (pilot report §5.2). The pilot report's post-hoc analysis puts almost all of that deficit in predicting the first base token of each next macro.
- On the larger held-out reserve (13.5M tokens) the pilot report gives 1.136, 1.199 and 1.238 with 95% bootstrap intervals of about ±0.004; those intervals cover sampling of the validation text, not seeds. They come from an analysis log attached to the Fraise run, not from these metrics.

## Limitations
- One seed per model; seed variance is unmeasured.
- Both transformer baselines kept untied input embeddings at N(0,1) and had no final LayerNorm. At 38M parameters, fixing both cut the 64k/64k model's gap to the 8k/8k from 0.152 to 0.085 BPB, and a 64k/64k on a standard BPE closed it, so this table overstates the cost of a 64k vocabulary. Fraise normalises its state before its decoder, so its comparison with the baselines is also affected.
- The macro models took 17% more optimizer steps, at a different batch and learning rate.
- Equal text is one axis; at equal transformer compute the pilot report finds the comparison not established.
- If this report is regenerated, the generated table replaces this summary.

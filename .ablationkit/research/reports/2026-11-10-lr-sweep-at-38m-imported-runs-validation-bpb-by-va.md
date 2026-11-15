<!-- ablationkit:record {"schema_version":1,"record_id":"rep-20261110-7d1283","project_id":"fraise","kind":"report","tags":["generated"],"links":[{"kind":"study","id":"lr-sweep-38m"},{"kind":"hypothesis","id":"hyp-20261110-76aef8"},{"kind":"task","id":"task-20261110-apoxgb"}],"status":"draft","title":"LR sweep at 38M (imported runs): validation BPB by variant","summary":"Imported results of the AdamW learning-rate sweep at 38M parameters and 50M base tokens: the 8k/8k transformer is best at 1e-3 (1.4087 bits per byte, BPB), while the 64k/64k transformer on the macro tokenizer (1.5602) and Fraise (64k in, 8k out; 1.5408) are best at 3e-3, the largest rate tried, so their optimum may lie higher. Fraise beats the 64k/64k model at all three rates (by 0.006 to 0.019), but both trail the 8k/8k by more than 0.13. One seed per run, imported from files, and the 8k/8k and 64k/64k runs carry the two training faults found later by the gap audit.","evidence":[{"ref":{"kind":"run","id":"ef798bbe064841f98bbcddd1fc79c1ae","label":"sweep-fraise-0.0003"},"role":"candidate","mlflow_run_id":"ef798bbe064841f98bbcddd1fc79c1ae","variant_id":"fraise-adamw-3e-4","values":{"validation/bits_per_byte":"1.6795432127463263"},"note":"imported from data/experiments/poc-fineweb/models/sweep-fraise-0.0003, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"91b7a7daea3c420f829647d4a9476229","label":"sweep-symmetric-0.0003"},"role":"candidate","mlflow_run_id":"91b7a7daea3c420f829647d4a9476229","variant_id":"symmetric-adamw-3e-4","values":{"validation/bits_per_byte":"1.6943722776632544"},"note":"imported from data/experiments/poc-fineweb/models/sweep-symmetric-0.0003, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"a822365225e44108ae478b9eb7299327","label":"sweep-base-0.0003"},"role":"candidate","mlflow_run_id":"a822365225e44108ae478b9eb7299327","variant_id":"base-adamw-3e-4","values":{"validation/bits_per_byte":"1.4395092963967804"},"note":"imported from data/experiments/poc-fineweb/models/sweep-base-0.0003, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"dcb5f6919284425b98658de104f59bc7","label":"sweep-fraise-0.001"},"role":"candidate","mlflow_run_id":"dcb5f6919284425b98658de104f59bc7","variant_id":"fraise-adamw-1e-3","values":{"validation/bits_per_byte":"1.5925852978670871"},"note":"imported from data/experiments/poc-fineweb/models/sweep-fraise-0.001, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"58f39911ac2b470d9f547dc449f301fb","label":"sweep-symmetric-0.001"},"role":"candidate","mlflow_run_id":"58f39911ac2b470d9f547dc449f301fb","variant_id":"symmetric-adamw-1e-3","values":{"validation/bits_per_byte":"1.5984911149493963"},"note":"imported from data/experiments/poc-fineweb/models/sweep-symmetric-0.001, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"44b7fac11bc04820aacd4c6bb7f891e8","label":"sweep-base-0.001"},"role":"baseline","mlflow_run_id":"44b7fac11bc04820aacd4c6bb7f891e8","variant_id":"base-adamw-1e-3","values":{"validation/bits_per_byte":"1.4087389558521102"},"note":"imported from data/experiments/poc-fineweb/models/sweep-base-0.001, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"d7724196ad2b439cb84762aad2a46268","label":"sweep-fraise-0.003"},"role":"candidate","mlflow_run_id":"d7724196ad2b439cb84762aad2a46268","variant_id":"fraise-adamw-3e-3","values":{"validation/bits_per_byte":"1.5407766707797912"},"note":"imported from data/experiments/poc-fineweb/models/sweep-fraise-0.003, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"71306939fc57417f8e1da4cf685981f6","label":"sweep-symmetric-0.003"},"role":"candidate","mlflow_run_id":"71306939fc57417f8e1da4cf685981f6","variant_id":"symmetric-adamw-3e-3","values":{"validation/bits_per_byte":"1.5602483482901124"},"note":"imported from data/experiments/poc-fineweb/models/sweep-symmetric-0.003, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"},{"ref":{"kind":"run","id":"07ee6d2584ef4cc59b00f56c84f24b42","label":"sweep-base-0.003"},"role":"candidate","mlflow_run_id":"07ee6d2584ef4cc59b00f56c84f24b42","variant_id":"base-adamw-3e-3","values":{"validation/bits_per_byte":"1.4127033875283672"},"note":"imported from data/experiments/poc-fineweb/models/sweep-base-0.003, not run under AblationKit (reported commit d0fc54ea7182)","captured_at":"2026-11-10T20:53:16.910Z"}],"generated":{"study_id":"lr-sweep-38m","baseline_variant":"base-adamw-1e-3","metric_keys":["validation/bits_per_byte"],"runs":9,"generated_at":"2026-11-10T20:53:16.911Z"},"created_at":"2026-11-10T20:53:16.911Z","updated_at":"2026-11-11T07:38:11.208Z","created_by":{"kind":"agent","id":"6830bb2a421e119b"},"updated_by":{"kind":"agent","id":"6ef314f6613932d5"},"revision":3} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# LR sweep at 38M (imported runs): validation BPB by variant

_ID `rep-20261110-7d1283` · status draft · tags generated · agent, 2026-11-10T20:53:16.911Z_

Links:

- study `lr-sweep-38m`
- hypothesis `hyp-20261110-76aef8`
- task `task-20261110-apoxgb`

## Evidence summary

Imported results of the AdamW learning-rate sweep at 38M parameters and 50M base tokens: the 8k/8k transformer is best at 1e-3 (1.4087 bits per byte, BPB), while the 64k/64k transformer on the macro tokenizer (1.5602) and Fraise (64k in, 8k out; 1.5408) are best at 3e-3, the largest rate tried, so their optimum may lie higher. Fraise beats the 64k/64k model at all three rates (by 0.006 to 0.019), but both trail the 8k/8k by more than 0.13. One seed per run, imported from files, and the 8k/8k and 64k/64k runs carry the two training faults found later by the gap audit.

## Evidence

| Role | Variant | Object | Attempt | Source | Values |
| --- | --- | --- | --- | --- | --- |
| candidate | fraise-adamw-3e-4 | run `ef798bbe064841f98bbcddd1fc79c1ae` |  |  | validation/bits_per_byte=1.6795432127463263 |
| candidate | symmetric-adamw-3e-4 | run `91b7a7daea3c420f829647d4a9476229` |  |  | validation/bits_per_byte=1.6943722776632544 |
| candidate | base-adamw-3e-4 | run `a822365225e44108ae478b9eb7299327` |  |  | validation/bits_per_byte=1.4395092963967804 |
| candidate | fraise-adamw-1e-3 | run `dcb5f6919284425b98658de104f59bc7` |  |  | validation/bits_per_byte=1.5925852978670871 |
| candidate | symmetric-adamw-1e-3 | run `58f39911ac2b470d9f547dc449f301fb` |  |  | validation/bits_per_byte=1.5984911149493963 |
| baseline | base-adamw-1e-3 | run `44b7fac11bc04820aacd4c6bb7f891e8` |  |  | validation/bits_per_byte=1.4087389558521102 |
| candidate | fraise-adamw-3e-3 | run `d7724196ad2b439cb84762aad2a46268` |  |  | validation/bits_per_byte=1.5407766707797912 |
| candidate | symmetric-adamw-3e-3 | run `71306939fc57417f8e1da4cf685981f6` |  |  | validation/bits_per_byte=1.5602483482901124 |
| candidate | base-adamw-3e-3 | run `07ee6d2584ef4cc59b00f56c84f24b42` |  |  | validation/bits_per_byte=1.4127033875283672 |

## Interpretation

<!-- ablationkit:body -->
## Setup
Generated from the 9 runs of the study "LR sweep at 38M": three AdamW learning rates (3e-3, 1e-3, 3e-4) for each of three models at 38M dense parameters (d=512, 12 layers, SwiGLU 1024), 50M FineWeb-Edu base tokens, batch 64 × 512 positions (1,526 steps for the 8k/8k, 891 for the macro models), Muon 0.02, warmup 50, seed 1337. Validation once at the end over 262,144 held-out base tokens. All nine runs were imported from files written before the project used AblationKit (commit d0fc54e); they were not run under it.

Bits per byte (BPB) is the summed negative log-likelihood of the held-out targets, in bits, divided by the UTF-8 bytes they cover, so it compares models with different tokenizers (lower is better):

$$
\mathrm{BPB} = \frac{1}{B \ln 2} \sum_{t} -\ln p(x_t \mid x_{<t})
$$

## Results
The generated table (latest logged value per run; deltas against the 8k/8k at 1e-3 are descriptive, no significance test):

| Variant | n | validation/bits_per_byte (↓) |
| --- | --- | --- |
| base-adamw-1e-3 (baseline) | 1 | 1.4087 |
| base-adamw-3e-3 | 1 | 1.4127 (Δ +0.0039644 worse) |
| base-adamw-3e-4 | 1 | 1.4395 (Δ +0.030770 worse) |
| fraise-adamw-1e-3 | 1 | 1.5926 (Δ +0.18385 worse) |
| fraise-adamw-3e-3 | 1 | 1.5408 (Δ +0.13204 worse) |
| fraise-adamw-3e-4 | 1 | 1.6795 (Δ +0.27080 worse) |
| symmetric-adamw-1e-3 | 1 | 1.5985 (Δ +0.18975 worse) |
| symmetric-adamw-3e-3 | 1 | 1.5602 (Δ +0.15151 worse) |
| symmetric-adamw-3e-4 | 1 | 1.6944 (Δ +0.28563 worse) |

Parameters that differ between variants: `adamw_learning_rate`, `architecture`, `base_vocab_size`, `local_dim`, `local_num_layers`, `local_swiglu_up_dim`, `macro_embedding`, `macro_residual_dim`, `max_macro_length`, `minibatch_size`, `tie_word_embeddings`, `tokenizer_sha256`, `vocab_size`.

The same numbers by model and rate:

| Model | 3e-3 | 1e-3 | 3e-4 |
| --- | ---: | ---: | ---: |
| 8k/8k transformer | 1.4127 | **1.4087** | 1.4395 |
| 64k/64k transformer on the macro tokenizer | **1.5602** | 1.5985 | 1.6944 |
| Fraise (64k in, 8k out) | **1.5408** | 1.5926 | 1.6795 |

## Interpretation
- The 8k/8k transformer is fairly flat between 1e-3 and 3e-3 (0.004 apart) and clearly worse at 3e-4.
- Both macro models improve all the way to 3e-3, the top of the grid, so a higher rate might be better still; the pilot used 3e-3 for them.
- Fraise is ahead of the 64k/64k model at every rate (0.019, 0.006 and 0.015), the opposite of the 100M pilot's final ordering. The 1e-3 difference is small enough to be noise.
- At this scale both macro models trail the 8k/8k by 0.13 to 0.15 BPB at their best rates.

## Limitations
- One seed per run.
- Minibatch differs: 16 for the two 3e-3 transformer runs (started before the runner switched to 64) and all three Fraise runs (out of memory at 64), 64 for the rest. A minibatch only splits the step into accumulated parts, so results are comparable across it.
- The 8k/8k and 64k/64k runs kept untied input embeddings at N(0,1) and had no final LayerNorm. The gap audit repeated two of them (1.4087 and 1.5609, same seed) and, with both faults fixed, cut the 64k/64k model's gap to the 8k/8k from 0.152 to 0.085. The best rates may move once those faults are fixed.
- If this report is regenerated, the generated table replaces this summary.

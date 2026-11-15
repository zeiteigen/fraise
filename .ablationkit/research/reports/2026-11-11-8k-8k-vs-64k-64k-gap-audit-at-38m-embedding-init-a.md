<!-- ablationkit:record {"schema_version":1,"record_id":"rep-20261111-eec443","project_id":"fraise","kind":"report","tags":["generated"],"links":[{"kind":"study","id":"vocab-gap-audit-38m"},{"kind":"hypothesis","id":"hyp-20261111-821fd1"},{"kind":"hypothesis","id":"hyp-20261111-ad88b7"},{"kind":"task","id":"task-20261111-ljvjsb"}],"status":"draft","title":"8k/8k vs 64k/64k gap audit at 38M: embedding init and final norm","summary":"Generated evidence table for the gap audit at 38M parameters: the validation bits per byte (BPB) of all eight runs, each against the 8k/8k transformer with its original settings (1.4087). The 64k/64k transformer on the macro tokenizer moves from 0.152 to 0.090 BPB behind that baseline as the two fixes are added, the batch-32 run changes nothing, and the 64k/64k on a standard BPE with both fixes is the best run at 1.4045. One seed per variant; the reasoned conclusion, with the gap measured against the 8k/8k with the same fixes, is the gap audit's final report.","evidence":[{"ref":{"kind":"run","id":"55dffb830ecc43eab1a9d7bff9923a8d","label":"gap-bpe64k-emb002-finalnorm"},"role":"candidate","mlflow_run_id":"55dffb830ecc43eab1a9d7bff9923a8d","attempt_id":"j-3a59c3dd241a-a1","job_id":"j-3a59c3dd241a","variant_id":"bpe64k-emb002-finalnorm","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.404466860745541"},"artifacts":["inputs/data/ak/configs/gap/model-symmetric-finalnorm.json","inputs/data/ak/configs/gap/calibrate_budgets.json","inputs/data/ak/configs/gap/bpe64k-lr3e-3-emb002.json","inputs/data/ak/configs/gap/code.diff","data/ak/events/gap-bpe64k-emb002-finalnorm.jsonl","data/ak/runs/gap-bpe64k-emb002-finalnorm/tokenizer.json","data/ak/runs/gap-bpe64k-emb002-finalnorm/model_meta.json","data/ak/runs/gap-bpe64k-emb002-finalnorm/training.log"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"a27bc5e0903640f881c6b9e42fe1d819","label":"gap-sym-emb002-finalnorm-b32"},"role":"candidate","mlflow_run_id":"a27bc5e0903640f881c6b9e42fe1d819","attempt_id":"j-571b76a48e4c-a1","job_id":"j-571b76a48e4c","variant_id":"sym-emb002-finalnorm-b32","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.499633618460794"},"artifacts":["inputs/data/ak/configs/gap/model-symmetric-finalnorm.json","inputs/data/ak/configs/gap/code.diff","inputs/data/ak/configs/gap/sym-lr3e-3-emb002-b32.json","data/ak/events/gap-sym-emb002-finalnorm-b32.jsonl","data/ak/runs/gap-sym-emb002-finalnorm-b32/training.log","data/ak/runs/gap-sym-emb002-finalnorm-b32/model_meta.json"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"66f89726c9c14bc694faf5da8ef418df","label":"gap-base-emb002-finalnorm"},"role":"candidate","mlflow_run_id":"66f89726c9c14bc694faf5da8ef418df","attempt_id":"j-14e690035d19-a1","job_id":"j-14e690035d19","variant_id":"base-emb002-finalnorm","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.4139389713407164"},"artifacts":["inputs/data/ak/configs/gap/code.diff","inputs/data/ak/configs/gap/base-lr1e-3-emb002.json","inputs/data/ak/configs/gap/model-base-finalnorm.json","data/ak/runs/gap-base-emb002-finalnorm/model_meta.json","data/ak/runs/gap-base-emb002-finalnorm/training.log","data/ak/events/gap-base-emb002-finalnorm.jsonl"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"d1b76fbccc634b9e8162c6e42007a055","label":"gap-sym-emb002-finalnorm"},"role":"candidate","mlflow_run_id":"d1b76fbccc634b9e8162c6e42007a055","attempt_id":"j-e84e007c0167-a1","job_id":"j-e84e007c0167","variant_id":"sym-emb002-finalnorm","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.499040612903656"},"artifacts":["inputs/data/ak/configs/gap/model-symmetric-finalnorm.json","inputs/data/ak/configs/gap/code.diff","inputs/data/ak/configs/gap/sym-lr3e-3-emb002.json","data/ak/runs/gap-sym-emb002-finalnorm/model_meta.json","data/ak/events/gap-sym-emb002-finalnorm.jsonl","data/ak/runs/gap-sym-emb002-finalnorm/training.log"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"bd1cfd7db3e94eb8a6e922a447d3ab8c","label":"gap-sym-emb002"},"role":"candidate","mlflow_run_id":"bd1cfd7db3e94eb8a6e922a447d3ab8c","attempt_id":"j-f7f16e3f6e5f-a1","job_id":"j-f7f16e3f6e5f","variant_id":"sym-emb002","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.51800902453196"},"artifacts":["inputs/data/ak/configs/gap/sym-lr3e-3-emb002.json","inputs/data/ak/configs/gap/model-symmetric.json","inputs/data/ak/configs/gap/code.diff","data/ak/runs/gap-sym-emb002/training.log","data/ak/runs/gap-sym-emb002/model_meta.json","data/ak/events/gap-sym-emb002.jsonl"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"602142eeb36c48938a742fbd73aeaf51","label":"gap-base-emb002"},"role":"candidate","mlflow_run_id":"602142eeb36c48938a742fbd73aeaf51","attempt_id":"j-22001c8509bf-a1","job_id":"j-22001c8509bf","variant_id":"base-emb002","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.4077191593280356"},"artifacts":["inputs/data/ak/configs/gap/model-base.json","inputs/data/ak/configs/gap/code.diff","inputs/data/ak/configs/gap/base-lr1e-3-emb002.json","data/ak/runs/gap-base-emb002/model_meta.json","data/ak/events/gap-base-emb002.jsonl","data/ak/runs/gap-base-emb002/training.log"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"a9c5f4be2ac344bca94247765977d72a","label":"gap-sym-default"},"role":"candidate","mlflow_run_id":"a9c5f4be2ac344bca94247765977d72a","attempt_id":"j-ebe95a20b207-a1","job_id":"j-ebe95a20b207","variant_id":"sym-default","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.560872912245556"},"artifacts":["inputs/data/ak/configs/gap/sym-lr3e-3.json","inputs/data/ak/configs/gap/model-symmetric.json","inputs/data/ak/configs/gap/code.diff","data/ak/events/gap-sym-default.jsonl","data/ak/runs/gap-sym-default/model_meta.json","data/ak/runs/gap-sym-default/training.log"],"captured_at":"2026-11-11T06:25:52.265Z"},{"ref":{"kind":"run","id":"4e102f8f5a714971a7880f1c6fe334b1","label":"gap-base-default"},"role":"baseline","mlflow_run_id":"4e102f8f5a714971a7880f1c6fe334b1","attempt_id":"j-f75668db86f6-a1","job_id":"j-f75668db86f6","variant_id":"base-default","source_commit":"72b1efb245cebdd2842485941a84df511766868b","source_dirty":true,"values":{"validation/bits_per_byte":"1.4086612371451699"},"artifacts":["inputs/data/ak/configs/gap/base-lr1e-3.json","inputs/data/ak/configs/gap/model-base.json","inputs/data/ak/configs/gap/code.diff","data/ak/runs/gap-base-default/model_meta.json","data/ak/runs/gap-base-default/training.log","data/ak/events/gap-base-default.jsonl"],"captured_at":"2026-11-11T06:25:52.265Z"}],"generated":{"study_id":"vocab-gap-audit-38m","baseline_variant":"base-default","metric_keys":["validation/bits_per_byte"],"runs":8,"generated_at":"2026-11-11T06:25:52.266Z"},"created_at":"2026-11-11T05:32:40.257Z","updated_at":"2026-11-11T07:38:11.195Z","created_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"updated_by":{"kind":"agent","id":"6ef314f6613932d5"},"revision":4} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# 8k/8k vs 64k/64k gap audit at 38M: embedding init and final norm

_ID `rep-20261111-eec443` · status draft · tags generated · agent, 2026-11-11T05:32:40.257Z_

Links:

- study `vocab-gap-audit-38m`
- hypothesis `hyp-20261111-821fd1`
- hypothesis `hyp-20261111-ad88b7`
- task `task-20261111-ljvjsb`

## Evidence summary

Generated evidence table for the gap audit at 38M parameters: the validation bits per byte (BPB) of all eight runs, each against the 8k/8k transformer with its original settings (1.4087). The 64k/64k transformer on the macro tokenizer moves from 0.152 to 0.090 BPB behind that baseline as the two fixes are added, the batch-32 run changes nothing, and the 64k/64k on a standard BPE with both fixes is the best run at 1.4045. One seed per variant; the reasoned conclusion, with the gap measured against the 8k/8k with the same fixes, is the gap audit's final report.

## Evidence

| Role | Variant | Object | Attempt | Source | Values |
| --- | --- | --- | --- | --- | --- |
| candidate | bpe64k-emb002-finalnorm | run `55dffb830ecc43eab1a9d7bff9923a8d` | j-3a59c3dd241a-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.404466860745541 |
| candidate | sym-emb002-finalnorm-b32 | run `a27bc5e0903640f881c6b9e42fe1d819` | j-571b76a48e4c-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.499633618460794 |
| candidate | base-emb002-finalnorm | run `66f89726c9c14bc694faf5da8ef418df` | j-14e690035d19-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.4139389713407164 |
| candidate | sym-emb002-finalnorm | run `d1b76fbccc634b9e8162c6e42007a055` | j-e84e007c0167-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.499040612903656 |
| candidate | sym-emb002 | run `bd1cfd7db3e94eb8a6e922a447d3ab8c` | j-f7f16e3f6e5f-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.51800902453196 |
| candidate | base-emb002 | run `602142eeb36c48938a742fbd73aeaf51` | j-22001c8509bf-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.4077191593280356 |
| candidate | sym-default | run `a9c5f4be2ac344bca94247765977d72a` | j-ebe95a20b207-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.560872912245556 |
| baseline | base-default | run `4e102f8f5a714971a7880f1c6fe334b1` | j-f75668db86f6-a1 | `72b1efb245ce` (dirty) | validation/bits_per_byte=1.4086612371451699 |

## Interpretation

<!-- ablationkit:body -->
## Setup
Generated from the 8 runs of the study "gap audit at 38M" (38M dense parameters, 50M FineWeb-Edu base tokens of text, one seed each); the baseline is the 8k/8k transformer with original settings. Values are the last logged validation bits per byte (BPB) of each run.

Bits per byte (BPB) is the summed negative log-likelihood of the held-out targets, in bits, divided by the UTF-8 bytes they cover, so it compares models with different tokenizers (lower is better):

$$
\mathrm{BPB} = \frac{1}{B \ln 2} \sum_{t} -\ln p(x_t \mid x_{<t})
$$

## Results
The generated table (deltas against the baseline; descriptive, no significance test):

| Variant | n | validation/bits_per_byte (↓) |
| --- | --- | --- |
| base-default (baseline) | 1 | 1.4087 |
| base-emb002 | 1 | 1.4077 (Δ -0.00094208 better) |
| base-emb002-finalnorm | 1 | 1.4139 (Δ +0.0052777 worse) |
| bpe64k-emb002-finalnorm | 1 | 1.4045 (Δ -0.0041944 better) |
| sym-default | 1 | 1.5609 (Δ +0.15221 worse) |
| sym-emb002 | 1 | 1.5180 (Δ +0.10935 worse) |
| sym-emb002-finalnorm | 1 | 1.4990 (Δ +0.090379 worse) |
| sym-emb002-finalnorm-b32 | 1 | 1.4996 (Δ +0.090972 worse) |

Variant names: `base` is the 8k/8k transformer, `sym` the 64k/64k transformer on the macro tokenizer, `bpe64k` the 64k/64k on a standard BPE; `emb002` is input embeddings initialised with standard deviation 0.02 instead of N(0,1), `finalnorm` a LayerNorm before the output head, `b32` batch 32 instead of 64.

## Interpretation
- Every delta here is against the *unfixed* 8k/8k. Measured against the 8k/8k with the same fixes, the 64k/64k macro model's gap is 0.152, 0.110 and 0.085 BPB, which is the series the final report uses; the +0.090 here is larger because the 8k/8k itself moved +0.005 with both fixes.
- The two fixes help the 64k/64k macro model a lot (−0.062 together) and the 8k/8k hardly at all (−0.001, then +0.006).
- Halving the batch (twice the steps) leaves the 64k/64k macro model where it was (+0.0006).
- The 64k/64k on a standard BPE with both fixes is the only 64k run level with or ahead of the 8k/8k, which points at the macro tokenizer rather than the vocabulary size.

## Limitations
One seed per variant, so differences under about 0.01 BPB are not resolved; 38M parameters only. The standard 64k BPE was trained on more in-domain text than the 8k tokenizer. If this report is regenerated, the generated table replaces this summary.

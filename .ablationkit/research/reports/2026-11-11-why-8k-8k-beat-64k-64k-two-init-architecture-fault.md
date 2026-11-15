<!-- ablationkit:record {"schema_version":1,"record_id":"rep-20261111-da2793","project_id":"fraise","kind":"report","tags":[],"links":[{"kind":"study","id":"vocab-gap-audit-38m"},{"kind":"hypothesis","id":"hyp-20261111-821fd1"},{"kind":"hypothesis","id":"hyp-20261111-ad88b7"},{"kind":"note","id":"note-20261111-b45691"},{"kind":"report","id":"rep-20261111-eec443"},{"kind":"task","id":"task-20261111-ljvjsb"},{"kind":"hypothesis","id":"hyp-20261111-c2ea9a"},{"kind":"hypothesis","id":"hyp-20261111-0f7adf"},{"kind":"note","id":"note-20261111-6a8f89"},{"kind":"job","id":"j-6ae5c7204172"}],"status":"final","title":"Why 8k/8k beat 64k/64k: two init/architecture faults plus the macro tokenizer, not vocabulary size","summary":"The gap audit's conclusion: at 38M parameters and 50M base tokens of text (one seed per run), the 0.152 bits-per-byte (BPB) gap between the 8k/8k transformer and the 64k/64k transformer on the macro tokenizer is explained without any bug in training or evaluation. Two initialisation and architecture faults, untied embeddings left at N(0,1) and no final LayerNorm, account for about 0.067 of it, step count for none, and the remaining 0.085 belongs to the cross-word macro tokenizer: a 64k/64k on a standard BPE with the same fixes scores 1.4045 BPB, matching or beating the 8k/8k (1.4077 to 1.4139). The 100M pilot has not been re-run with the fixes, and the standard BPE was trained on more in-domain text than the 8k tokenizer.","follow_up":"1. Re-run the pilot (100M, 500M tokens) for 8k/8k, 64k macro and 64k standard BPE with embedding_init_std 0.02 + final_norm (about 1.5 h each).\n2. A second seed of the 38M rows to bound noise.\n3. An 8k BPE trained on the same FineWeb-Edu slice, to remove the tokenizer-corpus confound.\n4. Re-tune AdamW LR for the fixed models.\n5. Consider word-bounded macro merges for Fraise, since cross-word merging is the remaining cost.\n6. Decide whether to make the two fixes defaults; they are opt-in and uncommitted (data/ak/configs/gap/code.diff).","evidence":[{"ref":{"kind":"run","id":"4e102f8f5a714971a7880f1c6fe334b1"},"role":"baseline","mlflow_run_id":"4e102f8f5a714971a7880f1c6fe334b1","values":{"validation/bits_per_byte":"1.4086612371451699"},"note":"8k/8k original settings","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"a9c5f4be2ac344bca94247765977d72a"},"role":"candidate","mlflow_run_id":"a9c5f4be2ac344bca94247765977d72a","values":{"validation/bits_per_byte":"1.560872912245556"},"note":"64k macro original settings","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"602142eeb36c48938a742fbd73aeaf51"},"role":"candidate","mlflow_run_id":"602142eeb36c48938a742fbd73aeaf51","values":{"validation/bits_per_byte":"1.4077191593280356"},"note":"8k/8k emb std 0.02","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"bd1cfd7db3e94eb8a6e922a447d3ab8c"},"role":"candidate","mlflow_run_id":"bd1cfd7db3e94eb8a6e922a447d3ab8c","values":{"validation/bits_per_byte":"1.51800902453196"},"note":"64k macro emb std 0.02","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"66f89726c9c14bc694faf5da8ef418df"},"role":"candidate","mlflow_run_id":"66f89726c9c14bc694faf5da8ef418df","values":{"validation/bits_per_byte":"1.4139389713407164"},"note":"8k/8k emb 0.02 + final norm","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"d1b76fbccc634b9e8162c6e42007a055"},"role":"candidate","mlflow_run_id":"d1b76fbccc634b9e8162c6e42007a055","values":{"validation/bits_per_byte":"1.499040612903656"},"note":"64k macro emb 0.02 + final norm","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"a27bc5e0903640f881c6b9e42fe1d819"},"role":"candidate","mlflow_run_id":"a27bc5e0903640f881c6b9e42fe1d819","values":{"validation/bits_per_byte":"1.499633618460794"},"note":"64k macro emb 0.02 + final norm, batch 32 (1,782 steps)","captured_at":"2026-11-11T06:18:41.827Z"},{"ref":{"kind":"run","id":"55dffb830ecc43eab1a9d7bff9923a8d"},"role":"candidate","mlflow_run_id":"55dffb830ecc43eab1a9d7bff9923a8d","values":{"validation/bits_per_byte":"1.404466860745541"},"note":"64k standard byte-level BPE (fraise-65536-456cef19bebbe6d6) emb 0.02 + final norm, budgets calibrated to the same text","captured_at":"2026-11-11T06:18:41.827Z"}],"created_at":"2026-11-11T06:18:41.828Z","updated_at":"2026-11-11T07:38:11.190Z","created_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"updated_by":{"kind":"agent","id":"6ef314f6613932d5"},"revision":3} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Why 8k/8k beat 64k/64k: two init/architecture faults plus the macro tokenizer, not vocabulary size

_ID `rep-20261111-da2793` · status final · agent, 2026-11-11T06:18:41.828Z_

Links:

- study `vocab-gap-audit-38m`
- hypothesis `hyp-20261111-821fd1`
- hypothesis `hyp-20261111-ad88b7`
- note `note-20261111-b45691`
- report `rep-20261111-eec443`
- task `task-20261111-ljvjsb`
- hypothesis `hyp-20261111-c2ea9a`
- hypothesis `hyp-20261111-0f7adf`
- note `note-20261111-6a8f89`
- job `j-6ae5c7204172`

## Evidence summary

The gap audit's conclusion: at 38M parameters and 50M base tokens of text (one seed per run), the 0.152 bits-per-byte (BPB) gap between the 8k/8k transformer and the 64k/64k transformer on the macro tokenizer is explained without any bug in training or evaluation. Two initialisation and architecture faults, untied embeddings left at N(0,1) and no final LayerNorm, account for about 0.067 of it, step count for none, and the remaining 0.085 belongs to the cross-word macro tokenizer: a 64k/64k on a standard BPE with the same fixes scores 1.4045 BPB, matching or beating the 8k/8k (1.4077 to 1.4139). The 100M pilot has not been re-run with the fixes, and the standard BPE was trained on more in-domain text than the 8k tokenizer.

## Evidence

| Role | Variant | Object | Attempt | Source | Values |
| --- | --- | --- | --- | --- | --- |
| baseline |  | run `4e102f8f5a714971a7880f1c6fe334b1` |  |  | validation/bits_per_byte=1.4086612371451699 |
| candidate |  | run `a9c5f4be2ac344bca94247765977d72a` |  |  | validation/bits_per_byte=1.560872912245556 |
| candidate |  | run `602142eeb36c48938a742fbd73aeaf51` |  |  | validation/bits_per_byte=1.4077191593280356 |
| candidate |  | run `bd1cfd7db3e94eb8a6e922a447d3ab8c` |  |  | validation/bits_per_byte=1.51800902453196 |
| candidate |  | run `66f89726c9c14bc694faf5da8ef418df` |  |  | validation/bits_per_byte=1.4139389713407164 |
| candidate |  | run `d1b76fbccc634b9e8162c6e42007a055` |  |  | validation/bits_per_byte=1.499040612903656 |
| candidate |  | run `a27bc5e0903640f881c6b9e42fe1d819` |  |  | validation/bits_per_byte=1.499633618460794 |
| candidate |  | run `55dffb830ecc43eab1a9d7bff9923a8d` |  |  | validation/bits_per_byte=1.404466860745541 |

## Follow-up

1. Re-run the pilot (100M, 500M tokens) for 8k/8k, 64k macro and 64k standard BPE with embedding_init_std 0.02 + final_norm (about 1.5 h each).
2. A second seed of the 38M rows to bound noise.
3. An 8k BPE trained on the same FineWeb-Edu slice, to remove the tokenizer-corpus confound.
4. Re-tune AdamW LR for the fixed models.
5. Consider word-bounded macro merges for Fraise, since cross-word merging is the remaining cost.
6. Decide whether to make the two fixes defaults; they are opt-in and uncommitted (data/ak/configs/gap/code.diff).

## Interpretation

<!-- ablationkit:body -->
## Setup
All runs: 38M dense (d=512, 12 layers), 50M FineWeb-Edu base tokens of text, batch 64×512 positions unless noted, AdamW 1e-3 (8k) / 3e-3 (64k), seed 1337, one seed per row. Validation covers the same held-out text in every row (971,115–971,133 bytes).

Bits per byte (BPB) is the summed negative log-likelihood of the held-out targets, in bits, divided by the UTF-8 bytes they cover, so it compares models with different tokenizers (lower is better):

$$
\mathrm{BPB} = \frac{1}{B \ln 2} \sum_{t} -\ln p(x_t \mid x_{<t})
$$

The gap is $\Delta = \mathrm{BPB}_{64\text{k}} - \mathrm{BPB}_{8\text{k}}$ for the same fixes.

## Results
| Run | BPB | Gap to 8k/8k (same fixes) |
|---|---:|---:|
| 8k/8k, original settings | 1.4087 | — |
| 64k/64k macro, original settings | 1.5609 | +0.152 |
| 8k/8k, emb std 0.02 | 1.4077 | — |
| 64k/64k macro, emb std 0.02 | 1.5180 | +0.110 |
| 8k/8k, emb 0.02 + final norm | 1.4139 | — |
| 64k/64k macro, emb 0.02 + final norm | 1.4990 | +0.085 |
| 64k/64k macro, emb 0.02 + final norm, batch 32 (1,782 steps vs 891) | 1.4996 | +0.086 |
| **64k/64k standard byte-level BPE**, emb 0.02 + final norm (1,186 steps) | **1.4045** | **−0.009** |

## Interpretation
### What the audit checked (the task's criteria)
1. **Training code audited:** no bug in loss, masking, packing, budgets or schedule (`note-20261111-b45691`). Two faults that hurt the large vocabulary:
   - untied embeddings left at PyTorch's N(0,1) init (`src/fraise/architectures/transformer.py:26`); the trained pilot embeddings keep cosine ≥0.93 to their init;
   - no final LayerNorm before lm_head (`src/fraise/model/model.py:48`).
   Neither is a crash-type bug; both are initialisation/architecture defects.
2. **Evaluation audited:** BPB is correct. Byte counts match across tokenizers, and the pilot report re-scored the checkpoints independently.
3. **Hyperparameters:**
   - Step count: no effect (batch 32 at 1,782 steps gives 1.4996 vs 1.4990).
   - Clipping: the fixes bring the 64k gradient norm under the clip (0.68–0.98 at the last step).
   - LR: not re-tuned after the fixes.
4. **Controlled runs:** 9 jobs, listed in the evidence.

### Attribution of the 0.152 BPB gap at this scale
- ~0.067 from the two faults (0.043 init, 0.019 final norm, net of an 8k change of +0.005);
- the remaining ~0.085 from the macro tokenizer, since a standard 64k BPE on the same text and with the same fixes erases it (−0.009);
- 0 from step count.

$$
\underbrace{0.152}_{\text{original gap}} \approx \underbrace{0.043}_{\text{embedding init}} + \underbrace{0.019}_{\text{final norm}} + \underbrace{0.005}_{\text{8k change}} + \underbrace{0.085}_{\text{macro tokenizer}}
$$

### Why the macro tokenizer costs this much
The 64k/64k pilot model uses `fraise-macro-tokenizer`: BPE merges over 8k token IDs with no word pre-tokenizer, only line-break boundaries. Its tokens cross words, it compresses harder (0.59 vs 0.777 tokens per 8k token for the standard 64k BPE), and 8.7% of its boundaries depend on text ≥2 tokens ahead (`data/ak/analysis/segmentation_lookahead.json`). The pilot report's per-slot analysis attributes the macro models' deficit mostly to END/boundary decisions, consistent with this.

## Limitations
- One seed per row; differences under ~0.01 BPB (e.g. 8k with vs without final norm, BPE 64k vs 8k) are not resolved.
- 38M / 50M tokens only; the pilot scale (100M / 500M) is not re-run.
- The 8k tokenizer was trained on a different, larger mix (BabyLM + Cosmopedia + FineWeb-Edu). The new 64k BPE was trained on the same 0.3 GB FineWeb-Edu slice as the macro tokenizer, so it is more in-domain than the 8k one.
- The standard-BPE run was made only with both fixes on.

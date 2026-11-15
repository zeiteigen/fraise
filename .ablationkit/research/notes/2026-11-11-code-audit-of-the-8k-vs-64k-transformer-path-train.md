<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261111-b45691","project_id":"fraise","kind":"note","tags":["vocab-gap","audit"],"links":[{"kind":"hypothesis","id":"hyp-20261111-821fd1"},{"kind":"hypothesis","id":"hyp-20261111-ad88b7"},{"kind":"task","id":"task-20261111-ljvjsb"}],"title":"Code audit of the 8k vs 64k transformer path (training, loss, evaluation)","created_at":"2026-11-11T04:37:54.135Z","updated_at":"2026-11-11T07:34:56.428Z","created_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":2,"summary":"A line-by-line audit of the training, loss and evaluation code of the 8k/8k transformer and the 64k/64k transformer on the macro tokenizer, made to find out whether a bug explains why the 64k model trails by 0.152 bits per byte (BPB) at 38M parameters. No bug was found: loss, masking, packing, budgets, the learning-rate schedule and BPB are computed the same, correct way for both, and both are scored on the same text (971,115 against 971,118 bytes in the pilot). It found two non-standard choices that weigh more on a 64k vocabulary, untied input embeddings left at N(0,1) initialisation and no final LayerNorm before the output head, which became the two hypotheses the gap audit tested, and a structural cost of the macro tokenizer: 8.7% of its token boundaries depend on text two or more base tokens ahead."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Code audit of the 8k vs 64k transformer path (training, loss, evaluation)

_ID `note-20261111-b45691` · tags vocab-gap, audit · agent, 2026-11-11T04:37:54.135Z_

Links:

- hypothesis `hyp-20261111-821fd1`
- hypothesis `hyp-20261111-ad88b7`
- task `task-20261111-ljvjsb`

A line-by-line audit of the training, loss and evaluation code of the 8k/8k transformer and the 64k/64k transformer on the macro tokenizer, made to find out whether a bug explains why the 64k model trails by 0.152 bits per byte (BPB) at 38M parameters. No bug was found: loss, masking, packing, budgets, the learning-rate schedule and BPB are computed the same, correct way for both, and both are scored on the same text (971,115 against 971,118 bytes in the pilot). It found two non-standard choices that weigh more on a 64k vocabulary, untied input embeddings left at N(0,1) initialisation and no final LayerNorm before the output head, which became the two hypotheses the gap audit tested, and a structural cost of the macro tokenizer: 8.7% of its token boundaries depend on text two or more base tokens ahead.

<!-- ablationkit:body -->
## Checked and found correct
- **Loss** (`training/batch.py:11-28`, `minibatches.py`): token-summed CE with ignore_index, divided by the batch's valid-target count; the minibatch weighting is exact. Macro targets are macro IDs; padding only in the final window, masked.
- **Packing** (`training/pipeline/packing.py`): identical code for both; budgets count base tokens via `unit_lengths`; the final macro window ends at the macro reaching the budget. Pilot step counts (15,259 / 17,896) match the planned budgets.
- **Schedule** (`runtime.py:118`): cosine to zero at the planned step count for both.
- **BPB** (`training/validation/evaluate.py`, `measures.py`): byte lengths of a macro = sum of its base tokens' byte-level lengths; specials have 0 bytes in both. Pilot final validation covers **971,115 bytes (base) vs 971,118 (symmetric)**, so both score the same text (from `model_meta.json`). The pilot report also re-scored the checkpoints to 4 decimals. No evaluation fault.
- **Macro tokenizer** (`tokenizer/macro.py`): expansion reproduces the base encoding; specials never merge.
- **Optimizer groups** (`optimizer.py`, `architectures/transformer.py:34-46`): Muon on transformer matrices; embedding and lm_head on AdamW with decay; biases and norms without decay. Same for both.

## Non-standard choices that load more on a 64k vocabulary
1. **Untied embeddings keep PyTorch's N(0,1) init** (`architectures/transformer.py:26-31` re-initialises only tied weights). With AdamW 1e-3/3e-3 the pilot embeddings end with cosine ≥0.93 to their init (`data/ak/analysis/embedding_drift.json`). → `hyp-20261111-821fd1`.
2. **No final LayerNorm before lm_head** (`model/model.py:48-51`). → `hyp-20261111-ad88b7`.
3. **Global grad clip 1.0 over all params** (`runtime.py:68`): symmetric is clipped on most pilot steps (norm 1.1–3.0); base only sometimes. Muon and Adam are roughly scale-invariant, so this is probably minor. Not tested.
4. **Fewer optimizer steps at equal text**: in the 38M sweep both models use batch 64×512 positions, so symmetric gets 891 steps against 1,526 for base. The pilot partly compensated with batch 32.
5. **AdamW LR at the edge of the grid for macro models** (sweep best 3e-3, the largest value tried).

## Tokenizer structure (not a bug, but a cost base 8k does not pay)
The macro BPE has no pre-tokenizer except line breaks, so merges cross words. `data/ak/analysis/segmentation_lookahead.py` on 27,974 validation macros: **8.7% of macro boundaries are decided only by base tokens ≥2 positions ahead** (7.7% at k=2, 0.9% at k≥3). A canonical-only scorer therefore charges the macro model for hedging between segmentations. This is consistent with the pilot report's END cost (0.056 BPB) and its 0.19 BPB heuristic estimate of the probability mass on consistent alternatives.

## Code changes made for testing (uncommitted, default off)
`ModelConfig.final_norm` and `TrainingSettings.embedding_init_std`, with a parameter-count update and tests (`tests/test_final_norm_and_embedding_init.py`, 7 tests). The full suite passes (221 + 7). Diff: `data/ak/configs/gap/code.diff`.

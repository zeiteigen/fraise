<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261111-a67f86","project_id":"fraise","kind":"note","tags":["vocab-gap","caveat"],"links":[{"kind":"study","id":"poc-pilot-100m"},{"kind":"study","id":"lr-sweep-38m"},{"kind":"hypothesis","id":"hyp-20261110-76aef8"},{"kind":"report","id":"rep-20261111-da2793"},{"kind":"task","id":"task-20261111-ljvjsb"}],"title":"Pilot and LR-sweep baselines were trained with untrained embeddings and no final norm","created_at":"2026-11-11T06:26:54.410Z","updated_at":"2026-11-11T07:34:56.439Z","created_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":2,"summary":"A warning on the two older studies: every 8k/8k and 64k/64k transformer run in the 100M pilot and the 38M learning-rate sweep was trained with untied input embeddings left at N(0,1) initialisation and without a final LayerNorm, the two faults the gap audit found. At 38M parameters and 50M tokens, fixing both moved the 64k/64k transformer on the macro tokenizer from 1.5609 to 1.4990 bits per byte (BPB) while the 8k/8k transformer stayed between 1.408 and 1.414, shrinking the gap from 0.152 to 0.085. The pilot's ranking (8k/8k 1.130 against 64k/64k 1.194) and any Fraise comparison drawn from it therefore overstate the cost of a 64k vocabulary; the pilot has not been re-run with the fixes."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Pilot and LR-sweep baselines were trained with untrained embeddings and no final norm

_ID `note-20261111-a67f86` · tags vocab-gap, caveat · agent, 2026-11-11T06:26:54.410Z_

Links:

- study `poc-pilot-100m`
- study `lr-sweep-38m`
- hypothesis `hyp-20261110-76aef8`
- report `rep-20261111-da2793`
- task `task-20261111-ljvjsb`

A warning on the two older studies: every 8k/8k and 64k/64k transformer run in the 100M pilot and the 38M learning-rate sweep was trained with untied input embeddings left at N(0,1) initialisation and without a final LayerNorm, the two faults the gap audit found. At 38M parameters and 50M tokens, fixing both moved the 64k/64k transformer on the macro tokenizer from 1.5609 to 1.4990 bits per byte (BPB) while the 8k/8k transformer stayed between 1.408 and 1.414, shrinking the gap from 0.152 to 0.085. The pilot's ranking (8k/8k 1.130 against 64k/64k 1.194) and any Fraise comparison drawn from it therefore overstate the cost of a 64k vocabulary; the pilot has not been re-run with the fixes.

<!-- ablationkit:body -->
Every transformer (8k/8k and 64k/64k) run in `poc-pilot-100m` and `lr-sweep-38m` has two settings that task `task-20261111-ljvjsb` found to be faults:
- untied embeddings left at PyTorch's N(0,1) init;
- no final LayerNorm before lm_head.

At 38M / 50M tokens, fixing both moved 64k/64k (macro) from 1.5609 to 1.4990 BPB, while 8k/8k stayed at 1.409–1.414. The gap fell from 0.152 to 0.085.

A standard word-pretokenized 64k BPE with the fixes scored 1.4045, so the remaining deficit belongs to the cross-word macro tokenizer (`rep-20261111-da2793`).

Consequently, the pilot's 8k/8k vs 64k/64k ranking (1.130 vs 1.194) and any Fraise-vs-baseline comparison drawn from it overstate the cost of a 64k vocabulary. The pilot has not been re-run with the fixes.

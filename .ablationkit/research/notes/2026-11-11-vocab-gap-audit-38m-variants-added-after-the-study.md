<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261111-6a8f89","project_id":"fraise","kind":"note","tags":["vocab-gap","protocol"],"links":[{"kind":"task","id":"task-20261111-ljvjsb"},{"kind":"study","id":"vocab-gap-audit-38m"},{"kind":"run","id":"55dffb830ecc43eab1a9d7bff9923a8d"},{"kind":"run","id":"a27bc5e0903640f881c6b9e42fe1d819"},{"kind":"job","id":"j-6ae5c7204172"}],"title":"vocab-gap-audit-38m: variants added after the study was created, and tokenizer provenance","created_at":"2026-11-11T06:26:44.305Z","updated_at":"2026-11-11T07:34:56.433Z","created_by":{"kind":"agent","id":"9ea139d6f718b0e8"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":2,"summary":"Provenance for the two runs added to the gap audit after it was created: the 64k/64k transformer on the macro tokenizer with both fixes at batch 32 (1.4996 bits per byte, BPB), and the 64k/64k on a standard BPE with both fixes (1.4045 BPB), including how the standard 64k tokenizer was trained and how its token budgets were calibrated so that it sees the same text. It also records that the planned 'final LayerNorm only' variant was never run. The caveat it carries: the new 64k BPE was trained on the same FineWeb-Edu slice as the macro tokenizer, while the 8k tokenizer was trained on a broader mix, so the 64k BPE is more in-domain."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# vocab-gap-audit-38m: variants added after the study was created, and tokenizer provenance

_ID `note-20261111-6a8f89` · tags vocab-gap, protocol · agent, 2026-11-11T06:26:44.305Z_

Links:

- task `task-20261111-ljvjsb`
- study `vocab-gap-audit-38m`
- run `55dffb830ecc43eab1a9d7bff9923a8d`
- run `a27bc5e0903640f881c6b9e42fe1d819`
- job `j-6ae5c7204172`

Provenance for the two runs added to the gap audit after it was created: the 64k/64k transformer on the macro tokenizer with both fixes at batch 32 (1.4996 bits per byte, BPB), and the 64k/64k on a standard BPE with both fixes (1.4045 BPB), including how the standard 64k tokenizer was trained and how its token budgets were calibrated so that it sees the same text. It also records that the planned 'final LayerNorm only' variant was never run. The caveat it carries: the new 64k BPE was trained on the same FineWeb-Edu slice as the macro tokenizer, while the 8k tokenizer was trained on a broader mix, so the 64k BPE is more in-domain.

<!-- ablationkit:body -->
The study's declared variant list cannot be edited after creation, so it lacks two variants that were run later. Both are tagged on their runs and appear in the regenerated report `rep-20261111-eec443`. The declared variant `sym-finalnorm` was **never run**.

- **`sym-emb002-finalnorm-b32`**: run `a27bc5e0903640f881c6b9e42fe1d819`, job `j-571b76a48e4c`, 1.4996 BPB.
  - 64k/64k macro, embedding std 0.02 + final LayerNorm, batch 32 (1,782 steps instead of 891).
  - Config `data/ak/configs/gap/sym-lr3e-3-emb002-b32.json`.
- **`bpe64k-emb002-finalnorm`**: run `55dffb830ecc43eab1a9d7bff9923a8d`, job `j-3a59c3dd241a`, 1.4045 BPB.
  - 64k/64k on a standard byte-level BPE, with embedding std 0.02 + final LayerNorm.
  - Tokenizer `data/experiments/poc-fineweb/tokenizers/fraise-65536-456cef19bebbe6d6.json`, made by job `j-6ae5c7204172`:
    - `create-tokenizer --vocab-size 65536 --sources fineweb-edu --source-size-gb 0.3`;
    - same trainer and word pre-tokenizer as the 8k;
    - same 0.3 GB FineWeb-Edu slice (validation prefix excluded) as the macro tokenizer.
  - Budgets count this tokenizer's own tokens, so they were calibrated to the same text with `data/ak/analysis/calibrate_budgets.py` (output `data/ak/configs/gap/calibrate_budgets.json`): 38,849,560 train and 203,610 validation tokens, at 0.777 tokens per 8k token (macro: 0.59).
  - Validation covered 971,133 bytes (8k: 971,115). The run selected its own training text from the same FineWeb-Edu stream order.
  - Config `data/ak/configs/gap/bpe64k-lr3e-3-emb002.json`, 1,186 steps.

Confound: the 8k tokenizer (`base-8192.json`) was trained earlier on a larger mix (BabyLM + Cosmopedia + FineWeb-Edu), so the new 64k BPE is more in-domain than the 8k.

## Update (2026-11-11)
Both variants are now declared on the study, with descriptions, and the never-run `sym-finalnorm` variant has been removed from its list. The facts above are unchanged.

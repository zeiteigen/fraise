# Level-2 (64k) tokenizer study

Candidate macro vocabularies over the 8k base tokenizer, compared at 38M parameters
(AblationKit study `level2-tokenizers-38m`). `colab.sh ARCH-TOKENIZER` runs one variant on a
fresh GPU runtime from a clean checkout; see its header for the variants and outputs.

| File | What it is |
| --- | --- |
| `tokenizers/bpe-8k.json` | The 8k base tokenizer every macro vocabulary is built on |
| `tokenizers/hierarchical-64k.json` | The original cross-word hierarchical BPE (merge replay) |
| `tokenizers/wordbpe-64k.json` | Word-bounded hierarchical BPE: no merge crosses a base pre-token |
| `tokenizers/greedy-cross-64k.json` | Prefix-closed vocabulary, longest-match segmentation, cross-word |
| `configs/` | Hierarchical-only 64k with both fixes; Fraise 64/8 at the sweep's settings |
| `tokenizer_stats.py` | Compression, multi-word share, unforced boundaries, lookahead, entry usage |

All macro tokenizers were trained with `create-tokenizer --source-size-gb 0.3` on the same
FineWeb-Edu slice; the two that `colab.sh` trains itself (`greedy-words`, `curriculum`) use the
same command with `--segmentation greedy --word-bounded` or `--word-bounded-vocab-size 58982`.
Validation timeouts are raised to one hour for slow runtime CPUs; nothing else differs from the
gap audit's configs.

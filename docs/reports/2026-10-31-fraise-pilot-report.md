# Fraise: architecture, first results, and open questions

*Report date: 2026-10-31. Code state: branch `t3/0e6f556d`, commit `d0fc54e`.
All training runs, the LR sweep and the pilot used that commit. The only exception
is the "chunked loss" bullet of the §5.3 benchmark, which ran on the unmerged
experimental branch `t3/chunked-loss` (commit `24dfcb8`).*

This document is written so that an engineer or AI agent with no other context can
understand what Fraise is, how it is implemented, what has been measured, and which
architectural decisions are open.
- Sections 1–6 report what was built and measured. Interpretations inside them are
  marked *(Interpretation:)*.
- **Section 7 is opinion** and is labelled as such throughout.
- Section 9 lists concrete next experiments.

---

## 0. Summary

- **Goal.** Test whether a language model can read a large *macro* vocabulary
  (64k tokens, each a fixed sequence of 1–8 base tokens) while writing a small
  *base* vocabulary (8k tokens). It should then spend fewer expensive global
  transformer steps per byte without losing access to the text inside tokens.
- **Built so far (PoC stages 1–6 of 9; list in §1).**
  - A macro tokenizer: BPE merges learned over base-token IDs, with exact
    decompositions.
  - Data pipeline budgets measured in base tokens, so all models see identical text.
  - Bits-per-byte (BPB) validation.
  - The `fraise` architecture: compositional macro input embeddings with 32-d
    learned residuals, a standard transformer over macro positions, and a local
    decoder that emits base tokens plus END, walked through a trie at generation
    time. The local decoder has 10.1M dense parameters plus a 6.3M output head.
    Together they cost ~258 MFLOP per position in training, about 41% of the
    transformer's own per-position FLOPs (§5.4). This 41% is unrelated to the 41%
    reduction in positions.
  - Baselines: an 8k/8k transformer on the base tokenizer, and a 64k/64k
    ("symmetric") transformer on the same macro tokenizer as Fraise.
- **Main result.** Held-out BPB after 500M FineWeb-Edu base tokens; lower is better.
  Each model has a 99.2M-parameter transformer; Fraise also has a 10.1M dense local
  decoder, so 109.3M dense in total. This is ~5 tokens per dense parameter, far below
  the PoC's target of ~40, so the runs are early in training (§8).

  | Model | BPB, training-time slice (262k tokens) | BPB, full held-out reserve (13.5M tokens; 95% CI) |
  |---|---|---|
  | Base 8k/8k | **1.130** | **1.136** [1.132, 1.140] |
  | Symmetric 64k/64k | 1.194 | 1.199 [1.195, 1.203] |
  | Fraise 64k-in/8k-out | 1.233 | 1.238 [1.234, 1.242] |

  Intervals cover validation-sample noise only; seed variance is unmeasured.

- **How the comparison depends on the compute axis** (details and caveats in §5.4):
  - *Equal tokens of text* (measured): base wins clearly.
  - *Equal total training FLOPs, or equal wall-clock* (estimated): base still wins,
    ~1.13–1.16 vs 1.194 / 1.233. The base model is better per token, and the macro
    models save less compute than their 41% fewer positions suggest, because the
    vocabulary heads and the local decoder are large at 100M parameters.
  - *Equal training positions* (equal transformer compute; the PoC lists "global
    transformer computation" as one tradeoff axis but names no single deciding
    axis; distinct from inference KV-cache size, see §5.4): **not established.** The base value at matching positions (~1.225) is read
    from partway through the base run, before its learning rate had decayed. A base run
    fully trained (annealed) to that budget would score better, so the apparent
    "symmetric beats base" (1.194) and "Fraise ≈ base" (1.233) are biased in the macro
    models' favour by an unknown amount. Experiment E0 (§9) narrows this; step-count
    and context-length confounds remain (§5.4, §8).
- **Fraise vs symmetric** (same tokenizer, data, batch, steps and transformer):
  - Fraise leads early (1.535 vs 1.674 at ~56M tokens).
  - Symmetric overtakes at ~208M tokens (interpolated) and finishes 0.039 BPB better.
    Post-hoc measurements on the final checkpoints (§5.5) localize it and put a
    sampling-noise interval on it.
- **Where Fraise loses (measured, §5.5).**
  - The gap to symmetric is +0.039 BPB. Its 95% CI on the full reserve,
    [+0.0386, +0.0395], covers validation-sample noise only. Seed variance is
    unmeasured, and at 38M dense / 50M tokens the sign was reversed (§5.1).
  - **≈99% net / ≈93% gross** of it comes from predicting the **first base token of
    each next macro** (slot 0).
  - Fraise is slightly worse on the second token and on END decisions, and better
    than symmetric from the third token of a macro onward.
  - Fraise's slot-0 deficit varies little with symmetric's estimated first-token
    entropy. Word-initial tokens carry 78% of it, at 1.2× the average deficit per
    decision.
- **Against base 8k (measured, §5.5e/i).**
  - Split by decision type, **slot-0 + END decisions carry 76% of Fraise's deficit to
    base and 60% of symmetric's**, and token decisions inside macros carry the rest.
  - The inside share also contains boundary-type cost: at most inside-macro token
    decisions the models put mass on END. The boundary-related share is therefore
    likely higher than 76% / 60%, by an unmeasured amount.
  - END decisions cost both macro models ~0.056 BPB.
  - Canonical-only scoring also charges the macro models for probability placed on
    *alternative segmentations of the same text*. How much this costs is unmeasured:
    a heuristic magnitude of ~0.19 BPB (§5.5i, neither a proven upper nor lower bound)
    exceeds the whole macro-vs-base gap, so
    under text-level (marginal) scoring the ranking against base is not known.
- **Open decisions for the project owner:** numeric success thresholds, a deployment
  setting (batch, context, memory) and a compute budget. None is given in `PoC.md`;
  §7.3 proposes thresholds.
- **Leading hypotheses (opinion, §7).**
  - Fraise's global states h_t are less informative, being trained only through the
    local decoder rather than a 64k-row head, and/or its slot-0 read-out path limits
    them, and/or slot 0 is simply miscalibrated (testable cheaply post hoc: D6).
  - A softmax-bottleneck explanation remains open (the evidence against it is weak).
  - An input-side cause is also open: composed embeddings, and residuals of rarely
    seen macros that stayed near their random init.
  - Optimization (an untuned LR at the grid edge, different clipping) is not ruled
    out.
  - Analyses of the frozen pilot weights separate read-out limits from state
    quality cheaply: the probes D1–D2, the attribution measurement D3 and the
    breakdown D5. D5 also gives an indirect, post-hoc test of input-side causes
    (B5); the definitive test needs retraining (E4). The LR re-sweep E6 tests
    optimization (§9).

---

## 1. Project goal and success criteria

"PoC" means proof of concept; `PoC.md` is the project's proof-of-concept
specification.

The project is specified in `PoC.md` (repository root). In short:

- **Asymmetric vocabularies.**
  - The global transformer reads macro tokens (64k vocabulary). Every macro has a
    deterministic decomposition into tokens of a base vocabulary (8k).
  - Output and generation happen in the base vocabulary.
  - The 8k output is deliberately small, so any gains cannot be explained by "bigger
    vocabulary everywhere".
- **Compositional macro representations.**
  - A macro's input vector is built from its constituent base-token embeddings, plus
    a small (32-d) learned macro-specific residual.
  - Design intent (untested): rare macros fall back on their parts; common ones can
    specialize.
- **Trie-based autoregression.**
  - A local decoder emits base tokens while walking a trie of macro decompositions.
  - When a complete macro is chosen, its macro ID is fed to the global transformer.
- **PoC implementation stages** (from `PoC.md`; each should be independently
  trainable and evaluable):
  1. 8k/8k baseline. *Done.*
  2. 64k/64k baseline. *Done (symmetric).*
  3. Static 64k-input / 8k-output Fraise. *Done.*
  4. Compositional macro embeddings. *Done* (implemented; whether it helps has not
     been ablated at pilot scale).
  5. 32-d learned macro residuals. *Done* (implemented; not ablated).
  6. Trie-based autoregressive bridging. *Done.*
  7. Multi-token prediction (MTP) in the local decoder: propose several base tokens
     between global steps. *Not built.*
  8. Dynamic decomposition: expand chosen macros into base tokens to give hard spans
     extra global positions and KV entries. *Not built.* `PoC.md` does not specify
     how spans are chosen (learned, heuristic or uncertainty-based).
  9. Variable-resolution / hierarchical KV-cache experiments. *Not built.*
- **Evaluation axes in the PoC.**
  - BPB (the primary tokenizer-independent metric).
  - Token-internal reasoning: spelling, morphology, numbers, unseen compositions.
    CharBench and CUTE are published benchmarks of character-level and orthographic
    understanding, such as spelling, character counting, and character
    insertion/deletion/swap within words. None of these evaluations exist in the
    codebase yet.
  - Efficiency: global positions per byte, KV entries per byte, throughput, VRAM, and
    a quality-vs-decomposition-compute curve.
- **Success criterion** (paraphrase of `PoC.md`). A *favourable tradeoff* between
  language-model quality, global transformer computation, sequence/KV length, and
  access to fine-grained token structure. The PoC does not require beating every
  baseline on raw BPB at equal tokens. **`PoC.md` gives no numeric thresholds** for
  any of these axes. It also specifies no deployment setting (batch size, context
  length, memory limits) and no total compute budget beyond the local single-GPU
  hardware. `PoC.md` is not reproduced here; this paraphrase is the only account of
  it in this document. Turning "favourable" into go/no-go numbers is an open decision for
  the project owner; §7.3 offers a proposal, labelled as opinion.
- **Training budget convention.** Budgets are measured in **base (8k) tokens**, so all
  architectures train on the same underlying text. The PoC target is ~40 base tokens
  per dense parameter on FineWeb-Edu.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| Base token / base vocabulary | Token of the 8,192-entry byte-level BPE tokenizer. All budgets and BPB are defined via base tokens and bytes. |
| Macro token / macro vocabulary | Token of the 65,536-entry macro tokenizer. IDs `0..8191` are the base tokens themselves; IDs `8192..65535` are learned merges with fixed base decompositions (length 2–8). |
| Decomposition | A macro's ordered base-token sequence. Concatenating the decompositions of a macro encoding reproduces the base encoding exactly. |
| Position / global position | One input slot of the global transformer (one macro for 64k models, one base token for 8k models). The KV cache holds one entry per position. |
| Slot (Fraise) | One local-decoder prediction: for the macro after position *t*, slots 0..k−1 predict its base tokens and slot *k* predicts END. |
| Units | Base tokens covered by targets; the budget measure for macro streams. |
| BPB | Bits per byte: summed NLL (in bits) of the held-out targets ÷ UTF-8 bytes those targets cover. |
| Symmetric | Plain transformer with 64k input **and** 64k output over the macro tokenizer. |
| Dense parameters | Matrices applied to every position regardless of token identity: the transformer, plus Fraise's local decoder blocks and W_c. Excludes embedding tables and output heads. |
| Lexical parameters | Rows selected or scored per token ID: embedding tables, Fraise's macro residuals and output heads. Fraise's small per-position gate and length tables (g, λ, g′, s) are counted with "base lexical" in the `params` breakdown, although they are indexed by position, not ID. |

---

## 3. System architecture

### 3.1 Base tokenizer

- Byte-level BPE (Hugging Face `tokenizers`) with 8,192 entries, including
  `<|pad|>`=0, `<|bos|>`=1, `<|eos|>`=2.
- It adds BOS/EOS once per document.
- Experiments used
  `fraise-8192-babylm-cosmopedia-khanacademy-cosmopedia-stories-fineweb-edu.json`,
  copied as `base-8192.json`.
- FineWeb-Edu held-out text averages about 3.69 bytes per base token.
- It predates this work. The repository's `create-tokenizer` excludes the 0.05 GB
  FineWeb validation prefix from tokenizer training. That this artifact was created
  that way is assumed from its name and metadata, not independently verified.

### 3.2 Hierarchical (macro) tokenizer (`src/fraise/tokenizer/macro.py`, `macro_training.py`)

#### 3.2.1 What "hierarchical" means here

Text is represented at three nested levels. Each level is a deterministic grouping of
the level below:

```text
level 2  macro tokens  (65,536)   [' strawberry']                      ← global transformer input
level 1  base tokens   (8,192)    [' str'] ['aw'] ['ber'] ['ry']       ← model output / local decoder
level 0  bytes         (256)      ' ' 's' 't' 'r' 'a' 'w' 'b' 'e' 'r' 'r' 'y'
```

- **Level 0 → 1** is the ordinary byte-level BPE base tokenizer (§3.1).
- **Level 1 → 2** is a *second* BPE, learned over sequences of base-token **IDs**,
  not over text. Its merges combine adjacent base tokens (or earlier macros) into
  macros.

Because the second level only ever groups whole base tokens, the macro tokenizer
**never re-segments text**. It takes the base tokenization exactly as it is and
decides which consecutive base tokens to group. This gives the key invariant:

    expand(macro_encode(text)) == base_encode(text)          # tested

where `expand` concatenates each macro's *decomposition* (its base-ID sequence).
Every macro has exactly one decomposition, of length 1–8.

#### 3.2.2 Worked example (the tokenizer used in the experiments)

Input: `"The strawberry fields were harvested in the United States of America.\nThey sold 1,250 kg."`

**Base encoding** (28 tokens, `token#id`):
```text
<|bos|>#1 The#778 ' str'#573 aw#1241 ber#719 ry#693 ' fields'#4654 ' were'#524
' harvest'#8154 ed#268 ' in'#292 ' the'#266 ' United'#3764 ' States'#4356 ' of'#296
' America'#4156 .#16 \n#201 They#2480 ' sold'#4004 ' 1'#665 ,#14 2#20 50#4589
' k'#436 g#73 .#16 <|eos|>#2
```

**Macro encoding** (16 tokens; 1.75× fewer positions):
```text
<|bos|>#1 The#778 ' strawberry'#41796 ' fields'#4654 ' were'#524 ' harvested'#30532
' in the United States'#8815 ' of America'#12860 '.\n'#8192 They#2480 ' sold'#4004
' 1,'#8731 '250'#44753 ' kg'#14286 .#16 <|eos|>#2
```

**Merge trees.** `#k` is an ID; *rank* is the merge's position in the learned merge
list, so macro ID = 8192 + rank.

| Macro | Merge tree | Decomposition (base IDs) |
|---|---|---|
| ' strawberry' #41796 (rank 33,604) | ((' str' + 'aw')→' straw'#15030 + ('ber' + 'ry')→'berry'#12162) | 573, 1241, 719, 693 |
| ' in the United States' #8815 (rank 623) | ((' in' + ' the')→#8195 + (' United' + ' States')→#8317) | 292, 266, 3764, 4356 |
| ' harvested' #30532 | ' harvest' + 'ed' | 8154, 268 |
| ' of America' #12860 | ' of' + ' America' | 296, 4156 |
| '.\n' #8192 (rank 0, the most frequent pair) | '.' + '\n' | 16, 201 |
| ' 1,' #8731 / '250' #44753 | ' 1' + ',' / '2' + '50' | 665, 14 / 20, 4589 |
| ' kg' #14286 | ' k' + 'g' | 436, 73 |

Observations an architect should know:
- **Macros span word boundaries.** `' in the United States'` is one token; early
  merges are mostly frequent function-word pairs (#8193 `' of the'`, #8194 `', and'`,
  #8200 `' on the'`).
- **Macros also complete words the base tokenizer split** (`' str'+'aw'+'ber'+'ry'`).
  They join *suffixes* (`' harvest'+'ed'`) and units (`' k'+'g'`).
- **The intermediate merge `' straw'` (#15030) is itself a macro,** and a prefix of
  `' strawberry'` in the trie. This is the structure trie decoding and prompt
  healing exploit.
- **Numbers are grouped by frequency, not by magnitude or digit position**
  (`' 1,'` + `'250'`). *(Interpretation:)* this matters for any numeric or
  token-internal evaluation.
- **The binary merge tree exists only implicitly** (the `merges` list). The model uses
  only the *flat* decomposition into base tokens. The tree is a candidate for
  multi-resolution decomposition (`[strawberry]` → `[straw][berry]` →
  `[s][t][r]…`), but nothing uses it yet. Note the deepest level reachable through
  the base vocabulary is base tokens, not characters.

#### 3.2.3 Training (`train_macro_tokenizer`)

1. Encode every training document with the base tokenizer, including `<|bos|>` and
   `<|eos|>`.
2. Map each base ID *i* to the single Unicode code point `U+F0000 + i` (Supplementary
   Private Use Area-A).
   - Each base token becomes one character, so off-the-shelf BPE machinery can
     operate on base-ID sequences.
   - This caps the base vocabulary at 65,534.
   - The documents are written one per line to a temporary file, then streamed into
     training.
3. Configure a *macro-stage* HF tokenizer:
   - **Special tokens:** the base vocabulary's added tokens plus the required
     `<|pad|>`, `<|bos|>`, `<|eos|>` are registered as special added tokens. Text is
     split around them, so they never merge and always stay single macros.
   - **Line-break boundaries:** every non-special base token whose text contains
     `"\n"` (in this vocabulary: `'\n'`, `' \n'`, and three newline-plus-indent
     tokens) is a boundary, applied by a `Split` pre-tokenizer with behaviour
     `merged_with_previous`. A boundary token *ends* a segment, so it can be the
     *last* element of a macro (`'.\n'`) but never sits inside one.
   - Segments are therefore runs of base tokens between line breaks or specials;
     merges are counted only inside segments. Besides the linguistic motivation,
     this keeps the BPE trainer's "words" short enough to be fast.
4. Run HF's Rust `BpeTrainer`:
   - `vocab_size = 65,536` total.
   - `initial_alphabet` = all non-special base characters, so every base token exists
     as a single-character token.
   - `min_frequency = 2`.
   - `max_token_length = max_macro_length + 1` (the trainer's limit is exclusive).
   - The trainer greedily merges the most frequent adjacent pair, counting within
     segments, until the vocabulary is full. It produces 57,344 merges.
5. Convert the learned merge list into **macro-ID pairs** in rank order, assigning
   IDs `8192 + rank`. Verify that every macro is at most `max_macro_length` long and
   that no token is duplicated.

The PoC's ID layout is therefore:
- **IDs 0–8,191 are exactly the base tokens.** A base token is its own length-1
  macro, which is why base IDs can be fed anywhere a macro ID is expected; generation
  relies on this.
- **IDs 8,192–65,535 are merges** in the order they were learned. A lower ID means an
  earlier merge, which is usually more frequent.

#### 3.2.4 Encoding and decoding

- **Encode:**
  1. Base-encode the text.
  2. Map IDs to characters.
  3. Split into segments (specials and line-break boundaries as above).
  4. Apply standard BPE inside each segment: repeatedly merge the adjacent pair with
     the **lowest rank** among pairs present in the merge table, until no listed
     pair remains.
  5. Map the resulting tokens to macro IDs.
  - This runs in Rust (`encode_batch` is parallel). It is deterministic, and it is
    exactly the procedure the trainer's merges imply.
  - Single-text `encode` also returns display `tokens` and character `offsets`
    (union of the constituent base-token offsets).
- **Decode:** expand macro IDs to base IDs, then base-decode.
- **Canonical vs non-canonical segmentations.**
  - Encoding always yields one *canonical* macro sequence per text.
  - Many other macro sequences spell the same base sequence (e.g.
    `[' straw'] ['berry']` vs `[' strawberry']`).
  - Models are trained only on canonical sequences. Fraise's generation can emit
    non-canonical sequences (it may END after `' straw'`); that is valid text but
    off the training distribution.
  - Reported NLL/BPB score only the canonical sequence, as is standard for BPE
    language models.

#### 3.2.5 Artifact and API

- **The artifact is one JSON file:**
  - `format: "fraise-macro-tokenizer"` and `format_version: 1`.
  - `base_tokenizer`: the *exact* serialized base tokenizer, embedded as a string.
  - `base_tokenizer_sha256`, `max_macro_length`, `boundary_tokens`.
  - `merges`: the list of `[left_id, right_id]` in rank order.
  - Decompositions are rebuilt from the merges at load time.
  - Fingerprint = SHA-256 of the serialized file, the same scheme as base
    tokenizers.
- **Loading:** `load_tokenizer` detects the format.
- **API:**
  - The object duck-types the HF `Tokenizer` methods Fraise uses: `encode`,
    `encode_batch`, `decode`, `get_vocab_size`, `token_to_id` (delegated to the base,
    so specials share IDs), and `to_str`.
  - It adds `expand(ids)`, `decomposition(id)`, `encode_base_ids(base_ids)`,
    `unit_lengths` (base tokens per macro ID), and `decompositions` (padded
    [V, max_len] array). It exposes the embedded `.base` tokenizer.
- **Data budgets:** the pipeline uses `unit_lengths` to count base tokens (§3.3) and
  `.base` to select downloads.
- **Model binding:** Fraise copies `decompositions` and `unit_lengths` into model
  buffers (§3.4).

#### 3.2.6 Invariants (unit-tested)

- `expand(encode(text).ids) == base.encode(text).ids`, and decode round-trips.
- Specials are always single tokens; base IDs are their own macros.
- No macro contains a line-break token except as its last element.
- Lengths never exceed `max_macro_length`; serialization round-trips byte-for-byte;
  tampering with recorded fields is rejected.

#### 3.2.7 Consequences and limitations for architecture decisions

- **No better base segmentation.** Because macros only group base tokens, the
  hierarchy can never fix a poor base segmentation (`' str'+'aw'` stays inside
  `' strawberry'`). It also cannot create a macro that starts or ends *inside* a
  base token.
- **Static segmentation.** The macro segmentation is static (no dropout, no
  alternatives). Dynamic decomposition (PoC stage 8) would have to choose expansions
  on top of it, for example by replacing selected macro IDs with their base IDs,
  which is always possible thanks to the ID layout.
- **No character level.** The finest level a Fraise model ever sees is base tokens.
  Character-level structure (spelling) is accessible only through base tokens
  (`' str'`, `'aw'`, …), not individual characters.
- **Merge vocabulary versus compute.** Merge frequency, not linguistic structure,
  decides macros (frequent phrases, number chunks, suffixed words). The 64k
  vocabulary buys 1.705 base tokens per position on FineWeb-Edu. A larger macro
  vocabulary or a longer `max_macro_length` would *(expected, untested)* increase compression but make
  macros rarer: 31.9% already appear fewer than 1,000 times in the pilot's 500M
  training tokens (§5.5c). Neither the vocabulary size nor `max_macro_length` has
  been swept.

#### 3.2.8 Statistics of the tokenizer used in the experiments

Trained on 300 MB of FineWeb-Edu, so in-domain.

- Macro-length histogram over the vocabulary (length: count):
  1: 8,192 · 2: 34,305 · 3: 17,240 · 4: 4,431 · 5: 1,012 · 6: 248 · 7: 74 · 8: 34.
- On held-out FineWeb-Edu:
  - **1.705 base tokens per macro** (41% fewer global positions).
  - 53% of positions are multi-token macros.
  - 2.705 local-decoder slots per position (macro length + END).
- Coverage of the 65,536 macros:
  - 64,801 appear in a 4.7M-macro held-out sample.
  - Frequencies in the actual pilot training split (all ~293M training macros) are in
    §5.5. The earlier estimate scaled the 4.7M-macro sample's counts by
    293M / 4.7M: 33% of macros below 1,000 occurrences, 1.9% below 100.
- The decomposition trie has 74,318 non-root nodes (74,319 including the empty
  root): 65,536 complete macros plus 8,782 internal prefixes that are not themselves
  macros. END is valid only at complete-macro nodes.

An earlier tokenizer trained on BabyLM (dialogue-heavy) gave only 1.31× on FineWeb,
against 1.705× for the in-domain tokenizer. That single comparison, confounded with
corpus size and type, suggests *(Interpretation:)* in-domain training of the macro
merges matters. BabyLM has since been removed as a source.

#### 3.2.9 How it was created, and trainer quirks handled in code

```sh
uv run create-tokenizer --base-tokenizer BASE.json --vocab-size 65536 \
  --max-macro-length 8 --sources fineweb-edu --source-size-gb 0.3
```

Training on about 300 MB of text takes about 4 minutes; artifacts are reused when the
same request is repeated.

- `BpeTrainer(max_token_length=n)` keeps only merges *shorter* than `n`; the code
  passes `max_macro_length + 1`.
- Calling `base.encode_batch` inside the iterator given to `train_from_iterator` can
  deadlock (nested work in Rayon, the Rust parallelism library the tokenizer uses, on the trainer's reader thread). Text is pre-encoded
  to a temporary file first.
- `Tokenizer.train(files)` keeps the `\n` line terminator as a symbol and can merge
  it. Training reads the file through an iterator that strips terminators.

### 3.3 Data pipeline and budgets

Paths are under `src/fraise/training/pipeline/` unless noted.

- **Sources.** `fineweb-edu` (HF `sample-10BT`, download cap 10.05 GB),
  `cosmopedia-khanacademy`, `cosmopedia-stories`. Held-out validation is the first
  0.05 GB of FineWeb-Edu. It is never used for training data or for the macro
  tokenizer; for the pre-existing base tokenizer this is assumed (§3.1).
- **Base-token budgets for every tokenizer.**
  - `dataset_token_counts`, `max_train_tokens` and `max_validation_tokens` count base
    tokens.
  - For macro streams, a target macro counts as its decomposition length ("units").
  - Downloads are selected using the *base* tokenizer (`counting_tokenizer`), so a
    macro run and an 8k run with equal budgets use byte-identical raw text and share
    the download cache.
- **Exact accounting.**
  - For macro streams, `plan_unit_windows` scans each source and finds how many
    packed windows are needed. The last window keeps the fewest targets that reach the
    budget, so a source can exceed its count by less than one macro. Observed:
    0 extra base tokens in the pilot (exactly 500,000,000) and +2 in the sweep
    (50,000,002).
  - Planning records the exact per-epoch units. Training checks them at epoch end,
    reports progress and throughput in base tokens, and resumes exactly (unit-aware
    window skipping).
- **Packing.** Documents are concatenated into one causal stream per source (BOS/EOS
  separate documents; attention is not blocked across them) and cut into windows of
  `sequence_length` *positions*. For 64k models that means macros, so a 512-position
  window covers about 873 base tokens.
- **Loss normalisation.** The training loss is the mean NLL per *target position*
  (per macro for 64k models, per base token for 8k). The Fraise per-macro NLL is
  directly comparable to symmetric's per-macro CE; neither is comparable to base's
  per-token loss.
- **BPB** (`src/fraise/training/validation/measures.py`, `evaluate.py`).
  - Byte lengths per base ID come from the byte-level vocabulary (one character per
    byte; specials 0 bytes). Macro byte length is the sum over its decomposition.
  - BPB = Σ NLL / ln 2 / Σ bytes over scored targets. Shown in the run log, in
    `model_meta.json → validation_results[*].bits_per_byte`, in the web inspector, and
    in `score_text`.
  - Each model scores its own windows over the same held-out prefix (262,144 base
    tokens in the pilot). EOS/BOS targets carry NLL but no bytes for every model.

### 3.4 Models

All three variants share the same global transformer implementation
(`src/fraise/model/transformer.py`):
- Pre-LayerNorm blocks.
- Multi-head attention with RoPE (base 10,000, applied to the full 64-d query/key
  per head), fused QKV, no biases.
- SwiGLU MLP with biases.
- **No final LayerNorm** in the baselines: `lm_head` reads the raw residual stream.
  Fraise applies a LayerNorm to h_t before its local decoder (`LN` in `W_c·LN(h_t)`
  below), and another before its head. This asymmetry is inherited from the
  pre-existing baseline code, was not ablated, and is listed in §8.
- PyTorch SDPA causal attention, KV cache for inference.
- No dropout anywhere.
- PyTorch default initializations unless stated otherwise:
  - `nn.Embedding` N(0,1).
  - `nn.Linear` Kaiming-uniform (U(±1/√fan_in) for weights and biases).
  - LayerNorm weight 1 / bias 0.

**8k/8k and 64k/64k baselines** (`architecture: "transformer"`, `src/fraise/model/model.py`)
- `token_embedding` (V×d, N(0,1)) → transformer → `lm_head` (Linear d→V with bias,
  default init), untied in all experiments. `tied_weight_init_std` applies only to
  tied models and was unused.
- Symmetric is the same model with V = 65,536 on macro inputs and macro targets.

**Fraise** (`architecture: "fraise"`, config `src/fraise/config/hierarchical.py`,
model `src/fraise/model/hierarchical.py`, adapter `src/fraise/architectures/hierarchical.py`)

*Buffers* (persistent, so checkpoints are self-contained):
- `decompositions` [V_macro, L] int64.
- `macro_lengths` [V_macro].
- Bound from the tokenizer at run start (`bind_tokenizer` hook); checked by
  `check_tokenizer`.

*Input embedding* `MacroEmbedding`, for macro *m* with parts b₁..b_k (k ≤ L = 8):

    x(m) = Σᵢ gᵢ ⊙ E[bᵢ] / √k  +  λ(k)  +  P · r(m)

- E is a base embedding table (8192×d, N(0,1) init).
- gᵢ are per-position gates (L×d, init 1), making the composition order-aware.
- λ is a length embedding (init 0).
- r(m) ∈ ℝ³² is a per-macro residual (65,536×32, N(0,1) init).
- P is a 32×d projection, **initialised to zero**, so every macro starts as its
  constituents.
- AdamW weight decay (0.01, decoupled) shrinks every row of the residual table,
  frequent or rare, by about 24% in total over the pilot: Σ lr·wd ≈ 0.003 × 0.01 ×
  ~8,950 effective steps under the cosine schedule ≈ 0.27, and e^−0.27 ≈ 0.76.
  Embedding gradients are dense (zero for absent rows), so Adam's momentum keeps
  moving rows for some steps after they were last used. *(Interpretation:)* if
  the intended "rare macros fall back on composition" effect happens at all, it comes
  from rare rows receiving few gradient updates, not from decay. It holds only if
  ‖P·r(m)‖ stays small for rare macros, which was not measured (see the next
  bullet).
- Possible failure mode (unmeasured): r(m) starts N(0,1) and P starts at 0. Once P is
  learned from frequent macros, the barely-updated random residuals of rare macros
  contribute P·r(m) as input noise. Weight decay shrinks them by only ~24%. The norm
  of P·r(m) by macro frequency was never inspected.
- Weight decay also applies to the 2-D gate tensors g (and g′, s_j, λ). It pulls them
  toward 0, not toward their initial value of 1 for g and g′. The size of this effect
  was not measured.
- Ablation switches:
  - `macro_embedding: "independent"` uses a plain V×d table and requires
    `macro_residual_dim: 0`.
  - `macro_residual_dim: 0` removes residuals.

*Global transformer.* Identical to the baselines, run over macro positions; it outputs
hidden states h_t.

*Local decoder* (`LocalDecoder`). For the macro m_{t+1} = (b₁..b_k), predict slot j ∈ [0, k]:

    z_j = W_c · LN(h_t)  +  Σ_{i<j} g'ᵢ ⊙ E'[bᵢ]  +  s_j
    u_j = Block_n(…Block_1(z_j))      Block(x) = x + SwiGLU(LN(x))
    logits_j = Head(LN(u_j)) ∈ ℝ^{8192+1}         # index 8192 = END
    target_j = b_{j+1} for j < k;  END for j = k

- `W_c`: d → local_dim.
- E′ is a separate local token embedding (8192×local_dim).
- g′ are local position gates ([L, local_dim], init 1).
- E′ is initialized N(0,1).
- λ(k) is a [L+1, d] table and s_j a [L+1, local_dim] table, both init 0.
- `local_dim` = 768 and `local_num_layers` = 2 were the author's untuned choices
  (local_dim set equal to d).
- Other choices and their origins:
  - from `PoC.md`: the 64k macro vocabulary and the 32-d residual dimension;
  - the author's own: `max_macro_length` = 8 and separate input (E) and local (E′)
    base tables. Neither was tuned.
- Scales at initialization: each prefix term g′ᵢ ⊙ E′[bᵢ] has per-component std ≈ 1,
  so the prefix sum at slot j has std ≈ √j. The context W_c·LN(h_t) has std ≈ 0.58
  (Kaiming-uniform over 768 unit-variance inputs). Trained scales were not measured
  (D3 in §9).
- Special tokens are ordinary length-1 macros. Predicting `<|eos|>` costs a slot-0
  token decision plus an END decision.
- Head: Linear(local_dim → 8,193) with bias, default init, **not** tied to E′ or E.
  The input base table E and the local table E′ are separate parameters; sharing
  them has not been tried.
- The prefix enters as a **gated sum**, not through attention. Every slot is
  independent given (h_t, prefix), so training evaluates all valid slots in one batch.
  The design was chosen for that parallelism and low cost. A local attention decoder
  was not tried. `local_swiglu_up_dim` 2048 matches the transformer's SwiGLU width;
  it was not tuned.
- A macro's NLL = Σ_j CE(logits_j, target_j), i.e. −log P(base tokens, then END).
- Training does **not** mask logits to the trie. The unmasked distribution is a valid
  sub-distribution, so the reported NLL is an upper bound on the trie-masked NLL.

*Training loss path.*
- `target_loss_sum(input_ids, targets, mask)` computes h, the context, and prefix
  cumsums on a padded [B, T, L+1] grid (fixed shape, compiled).
- The valid slots (`slot ≤ length`) are selected with `nonzero()`, and the
  MLP/head/CE run only on those rows (~2.7 per position) in
  `_valid_slot_loss`, which is `@torch.compiler.disable`.
- The variable row count would otherwise force one recompilation per distinct count.
  That tail is dominated by one GEMM, and benchmarks showed no gain from compiling it.
- Cost: one host sync per minibatch.
- `fraise.training.batch.token_loss_sum` dispatches to any model method named
  `target_loss_sum`.

*Generation* (`src/fraise/inference/hierarchical.py`).
- Base tokens are sampled only among the children of the current trie node, plus END
  if the node spells a complete macro. Temperature, top-k and top-p apply within that
  set. A node with one option skips the decoder.
- Root excludes PAD and BOS.
- Completed macro IDs are appended to the global context. The KV cache advances one
  entry per macro; when context exceeds `max_seq_len` it falls back to recomputing
  the sliding window.
- **Prompt healing:** the final prompt macro is removed and its base tokens are forced
  as the first trie path, so a prompt ending in "straw" can continue as "strawberry".
- Generation ids and counts are base tokens.
- `benchmark_generation` reports `global_steps_per_token`.

*Optimizer layout* (identical code for all architectures; `training/optimizer.py`).
- **Muon** for hidden 2-D matrices: all transformer matrices, plus Fraise's local
  SwiGLU matrices and `W_c`.
  - Settings: lr 0.02, momentum 0.95, Nesterov, 5 quintic Newton–Schulz steps in
    BF16.
  - Q, K and V are orthogonalized as separate blocks.
  - Per-matrix step size lr·√max(1, rows/cols).
  - Decoupled weight decay: each step multiplies weights by (1 − lr_t·0.01), with
    lr_t the scheduled Muon LR (peak 0.02), *not* the per-matrix shape-scaled step
    size.
    - Over a full cosine schedule this compounds to ≈ e^(−0.02·0.01·N/2): ≈0.22× for
      base (N = 15,259 steps) and ≈0.17× for the macro pilot runs (N = 17,896).
    - The transformer matrices therefore receive substantial and model-dependent
      total decay.
    - This decay pull is balanced by updates; it is not the final weight scale. The
      difference between the runs is a confound of the step-count choice.
- **AdamW** for everything else: embeddings, residual table, P, gates, output heads.
  - Settings: lr per config, betas (0.9, 0.95), ε 1e-8 (torch default),
    weight decay 0.01 on ≥2-D tensors.
  - Biases and norm parameters are not decayed.
- **Schedule.** One HF `get_cosine_schedule_with_warmup` drives both optimizers:
  linear warmup over `warmup_steps`, then cosine decay to **zero** at the final
  planned step. Muon and AdamW share the multiplier.
- **Gradient clipping.** The global grad-norm clip of 1.0 is applied before both
  optimizers.

**Parameter counts** (pilot configs: d=768, 14 layers, 12 heads × 64, SwiGLU 2048, context 512)

| Group | Base 8k | Symmetric 64k | Fraise |
|---|---:|---:|---:|
| Transformer (dense) | 99.2M | 99.2M | 99.2M |
| Local decoder (dense) | – | – | 10.1M |
| Base lexical (E, E′, gates…) | 6.3M (input table) | – | 12.6M |
| Macro lexical (per-macro rows) | – | 50.3M (input table) | 2.1M (32-d residuals + P) |
| Output head | 6.3M | 50.4M | 6.3M |
| **Total** | **111.8M** | **199.9M** | **130.3M** |

(`uv run params CONFIG.json` prints these breakdowns for Fraise.)

### 3.5 Code map (where to change things)

| Concern | Files |
|---|---|
| Macro tokenizer, training, artifact | `tokenizer/macro.py`, `tokenizer/macro_training.py`, `tokenizer/artifacts.py`, `cli/create_tokenizer.py` |
| Unit budgets, window planning | `training/pipeline/packing.py` (`plan_unit_windows`, `_iter_unit_blocks`), `budgets.py` (`plan_epochs`), `preparation.py`, `tokenization.py` |
| Training loop and state | `training/training.py`, `run/state.py` (`epoch_loss_targets` vs `epoch_valid_tokens`), `runtime.py` (`bind_tokenizer` call) |
| BPB | `training/validation/measures.py`, `evaluate.py` (`evaluate_scores`), `queue.py` |
| Fraise model | `config/hierarchical.py`, `model/hierarchical.py`, `architectures/hierarchical.py` |
| Generation and scoring | `inference/hierarchical.py`, `streaming.py`, `scoring.py`, `benchmark.py` |
| Architecture contract | `architectures/contracts.py` (optional `check_tokenizer`, `bind_tokenizer`, `parameter_breakdown`); `docs/architectures.md` |
| Tests | `tests/test_macro_*.py`, `tests/test_fraise_*.py` (loss equals step-by-step decoding; exact resume; trie validity; unit packing vs brute force) |

---

## 4. Experimental setup

The exact model configs, training hyperparameters (as supplied and fully
normalized), tokenizer checksums and execution provenance for every run are in
**Appendix A**.

- **Hardware.** One RTX 5090 Laptop GPU (24 GB), 32 GB RAM. BF16 autocast,
  `torch.compile` for the fixed-shape forward and loss. Validation runs on CPU in a
  separate process (FP32), in parallel with training.
- **Data.** FineWeb-Edu `sample-10BT`, in stream order. `shuffle_per_epoch: false`
  keeps document order. `shuffle_data: true` would randomly interleave packed windows
  *across sources*; with a single active source it has no effect. Validation is 262,144 base tokens of the
  held-out prefix.
- **Experiment directory.** `data/experiments/poc-fineweb/` (relative to the main
  checkout, `/home/munch/Projects/fraise`):
  - `configs/`, `logs/`, `models/`.
  - `raw/` is a symlink to the shared `data/raw/`.
  - Sweep runs: `sweep-{base,symmetric,fraise}-{0.003,0.001,0.0003}`.
  - Pilot runs: `pilot-{fraise,symmetric,base}`.
  - Runner scripts: `sweep.sh` (first sweep runner), `sweep2.sh` (its replacement
    mid-sweep, switching to minibatch 64 with an out-of-memory fallback to 16),
    `pilot.sh`, `prepare.py` (download and tokenize), `bench.py` (§5.3 benchmark).
- **LR sweep.**
  - 38M-dense models (d=512, 12 layers, 12 heads × 64 = 768 attention width, SwiGLU
    1024: 37.8M transformer parameters). Totals with embeddings and heads: base
    46.2M, symmetric 105M, Fraise 55.9M (transformer 37.8M + local decoder 3.4M +
    base lexical 8.4M + macro lexical 2.1M + head 4.2M). Earlier drafts called these
    "42M". The attention width ≠ d choice is inherited from the repository's example
    config.
  - 50M base tokens each.
  - Batch 64 × 512 positions; Muon 0.02; AdamW lr ∈ {3e-3, 1e-3, 3e-4}; warmup 50.
  - All architectures used the same batch in positions. The macro models therefore
    saw about 56k base tokens per step against 32k for base 8k, giving 891 vs 1,526
    steps.
  - Compression on the sweep's first 50M tokens was 1.712 base tokens per position
    (50M ÷ 891 steps ÷ 32,768 positions), vs 1.705 on held-out text. It varies
    slightly with the text.
  - Steady throughput at this size (base tokens/s): base ~162–165k, symmetric
    ~188–191k, Fraise ~146–150k. A 300M-token run therefore takes about 31 / 27 /
    34 min plus ~2 min of final validation.
- **Pilot.**
  - 100M-dense models, 500M base tokens each.
  - Base 8k: batch 64 × 512, AdamW 1e-3.
  - Symmetric and Fraise: batch **32** × 512, AdamW 3e-3. This is ~27.9k base tokens
    per step, so they get **17% more optimizer steps** than base (17,896 vs 15,259).
  - Why batch 32: the project owner chose it deliberately. 64 ÷ 1.705 ≈ 38 sequences
    would match base's base tokens per step; 32 gives the macro models slightly more
    steps. The owner's stated reason, "to account for a more complex architecture",
    is an untested belief, not a measured effect. It was not a memory workaround:
    accumulation would have allowed batch 64. The AdamW LR was **not** re-tuned for
    batch 32.
  - Muon 0.02, warmup 200, weight decay 0.01, gradient clip 1.0.
  - Accumulation: none for the macro models (minibatch 32 = batch 32). Base uses
    minibatch 32 with batch 64, i.e. two accumulated minibatches per step, which is
    mathematically equivalent to one 64-sequence step. Gradients are weighted by
    valid targets (macros for the macro models, base tokens for base).
  - Validation every 2,000 steps.
  - One seed per configuration; seed variance has not been measured.

---

## 5. Results

### 5.1 LR sweep (38M dense, 50M tokens; BPB)

| | AdamW 3e-3 | 1e-3 | 3e-4 |
|---|---|---|---|
| Base 8k (1,526 steps) | 1.413 | **1.409** | 1.440 |
| Symmetric 64k (891 steps) | **1.560** | 1.598 | 1.694 |
| Fraise (891 steps) | **1.541** | 1.593 | 1.680 |

- Fraise was ahead of symmetric at all three learning rates (by 0.019 / 0.005 /
  0.014). The 1e-3 difference is small enough to be noise; no variance estimate
  exists for the sweep.
- Both macro models preferred the highest learning rate tested. Their optimum may lie
  above 3e-3, because the grid had no higher point.

### 5.2 Pilot (100M dense, 500M tokens) — validation BPB by step

Tokens seen = step × 32,768 (base) or step × ~27,935 (macro models). Rows at the same
step therefore compare different token counts; the token columns make this explicit.

| Step | Base tokens seen (base 8k) | Base 8k BPB | Base tokens seen (macro models) | Symmetric 64k BPB | Fraise BPB |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 66M | 1.455 | 56M | 1.674 | 1.535 |
| 4,000 | 131M | 1.362 | 112M | 1.521 | 1.462 |
| 6,000 | 197M | 1.315 | 168M | 1.446 | 1.430 |
| 8,000 | 262M | 1.247 | 224M | 1.381 | 1.387 |
| 10,000 | 328M | 1.200 | 280M | 1.327 | 1.346 |
| 12,000 | 393M | 1.156 | 335M | 1.274 | 1.296 |
| 14,000 | 459M | 1.133 | 391M | 1.232 | 1.260 |
| 15,259 (final, base) | 500M | **1.130** | | | |
| 16,000 | | | 447M | 1.202 | 1.238 |
| 17,896 (final, macro) | | | 500M | **1.194** | **1.233** |

**Fraise minus symmetric over training:** −0.139 (step 2k) → −0.059 (4k) →
−0.016 (6k) → +0.006 (8k) → +0.019 (10k) → +0.022 (12k) → +0.028 (14k) → +0.036 (16k)
→ **+0.039** (final). Linear interpolation puts the sign flip at step ≈7,450
(≈208M tokens). The gap then widens through the end of the run, while the
learning rate decays (cosine to zero at step 17,896), so the widening cannot be
separated from the schedule. Successive checkpoints of one run are strongly
correlated, so *(Interpretation:)* the monotone trend alone is weak evidence of a real gap. §5.5 gives
a bootstrap interval on the final checkpoints instead.

**Final training-loss snapshot** (per target, nats; not comparable across tokenizers):
- Fraise 5.32 per macro, symmetric 5.13 per macro (validation 5.38 vs 5.21 per macro).
- Base 2.84–2.89 per base token.

**Stability.**
- No loss spikes.
- Gradient norm (before clipping at 1.0):
  - Fraise: 0.5–0.67 at every logged step (logged every 200 steps), so it was
    presumably *(Interpretation:)* rarely if ever clipped.
  - Symmetric: ~3.0 at step 5k, 1.1–1.8 later, so its updates were clipped
    (scaled down) on most steps.
  - Base: 0.7–1.2 (clipped some of the time).
  - The effect of clipping on the comparison was not studied.
    - *(Interpretation:)* Muon's orthogonalized update is invariant to gradient
      scale, and Adam's update
      is roughly invariant to a *consistent* scale. Clipping therefore mainly
      reweights steps relative to each other inside the momentum and moment
      averages, rather than acting as an LR reduction.
  - Gradient norms are **not comparable across models**: the losses differ (per-macro
    NLL for the macro models, per-base-token CE for base), and so do the parameter
    sets.

### 5.3 Throughput and memory (pilot, steady state)

| | Base tokens/s | Peak memory | Wall-clock for 500M |
|---|---:|---:|---:|
| Base 8k (64×512, minibatch 32) | ~81k | 9.2 GB | ~1.71 h |
| Symmetric 64k (32×512) | ~94k | 13.4 GB | ~1.48 h |
| Fraise (32×512) | ~83k | 16.2 GB | ~1.67 h |

Throughput values are steady-state readings from the pilot training logs. The
minibatch and chunked-loss figures below come from `bench.py` (10 timed steps after
3 warm-up steps). The "~6%" eager-tail figure came from an earlier isolated benchmark
of 40 steps at 42M total parameters, comparing compiled with eager execution of the
tail.

Wall-clock is 500M ÷ steady throughput: training only, excluding compilation (25–35 s)
and the final validation (3–4 min).

Minibatch benchmark (100M dense, 64-sequence batch, steady steps):
- Speed is flat across minibatch sizes for each model where they fit: 16/32/64 for
  base, 16/32 for the macro models.
- Symmetric and Fraise run out of memory at minibatch 64.
- An experimental chunked output cross-entropy (branch `t3/chunked-loss`) was 2–13%
  slower. It did not reduce memory for the compiled baselines (Inductor, `torch.compile`'s code generator, already fuses
  their CE); it saved ~2 GB for Fraise. It is **not merged**.

Two things are unexplained (not profiled):
- **Fraise is barely faster than base** (+2.5%) despite ~22% fewer estimated FLOPs
  per base token, while symmetric's ~18% FLOP saving shows up as +16% speed.
  Possible causes:
  - the uncompiled local tail (~6% in an isolated benchmark);
  - one host sync per minibatch from `nonzero()`;
  - many small kernels for gathers, cumsum and padding on the [B, T, 9, d] slot grid;
  - lower GEMM efficiency on ~44k-row local matmuls.
- **Fraise's peak memory (16.2 GB) exceeds symmetric's (13.4 GB)** despite a much
  smaller head. Likely contributors (unverified): the padded [B, T, L+1, local_dim]
  prefix-sum tensors and their autograd copies, plus two local-block activation sets
  on ~2.7 rows per position.

*(Recommendation:)* profile before predicting cost at scale or for deeper local
decoders. Also unexplained: the compositional-embedding gathers and the padded
[B, T, 9, d] prefix grid are not in the §5.4 FLOP count.

### 5.4 Compute accounting: three axes, three answers

Analytic estimate of training FLOPs per *base token*:
- 6 × parameters per position for matmuls, plus causal attention (~33 MFLOP per
  position).
- Divided by 1.705 positions per base token for the macro models.
- Fraise counts 2.705 local slots per position, each using the two local blocks and
  the 8,193-way head.

| | Train FLOPs per base token | Global positions per base token | Measured base tokens/s |
|---|---:|---:|---:|
| Base 8k | ~666M | 1.00 | ~81k |
| Symmetric 64k | ~546M | 0.59 | ~94k |
| Fraise | ~520M | 0.59 | ~83k |

Each macro model's 500M-token result, against base 8k at the same budget on each
axis. "≤" and "worse by ≥" below express the *expected* direction of the annealing
bias, not a proven bound. The values carry ±0.015 absolute sampling noise. Because
all models are scored on the same text, the noise on cross-model *differences* is
smaller. The only paired interval measured is Fraise−symmetric on Slice A,
≈±0.002 (§5.5a). No paired interval exists for base-vs-macro differences.

Base values are read from the **single base run's validation curve**
(interpolated between checkpoints). They are not separately trained runs.

| Matching axis | Base-equivalent tokens | Base BPB there | Symmetric (1.194) | Fraise (1.233) |
|---|---:|---:|---|---|
| Equal tokens of text | 500M | 1.130 (measured, end of schedule) | worse by 0.064 | worse by 0.103 |
| Equal total training FLOPs | sym ~410M / Fraise ~391M | ≤ ~1.150 / ≤ ~1.157 | worse by ≥ 0.044 | worse by ≥ 0.076 |
| Equal wall-clock (current implementation; see §5.3) | sym ~431M / Fraise ~488M | ≤ ~1.143 / ≤ ~1.131 | worse by ≥ 0.051 | worse by ≥ 0.102 |
| Equal *training* global positions (transformer-only FLOPs) | ~293M | ≤ ~1.225 | **undetermined** (read as −0.031) | **undetermined** (read as +0.008) |

Two different things are often both called "positions":
- **Training cost.** Matching training positions (the last row) compares models
  trained with the same transformer compute.
- **Inference memory.** KV-cache entries per byte at inference are a property of the
  tokenizer, not of the training budget. Any macro model uses ~0.59× as many KV
  entries and global decode steps per byte as any base-8k model (1 / 1.705),
  regardless of how either was trained, for canonical segmentations. In generation,
  non-canonical segmentations raise it slightly: Fraise measured 0.603 global steps
  per base token against the canonical 0.587 (§5.5g/h). That advantage is factual but says nothing
  about quality.

**Why base values are expected upper bounds ("≤").**
- Points before the end of the base run were taken while its learning rate was still
  high: about 0.37× peak at step 8,950, the ~293M-token point (cosine multiplier 0.5·(1+cos(π·(8,950−200)/(15,259−200))).
- A base model trained with a schedule that *ends* at that budget (annealed) would
  score better. The size of that effect has not been measured here, and no external
  figure is relied on. The base run's own curve flattens from 1.133 (step 14,000) to
  1.130 (15,259) as the LR approaches 0, but that is not an estimate of the annealing
  gain at ~293M tokens.
- The interpolated values come from the training-time validation slice ("Slice A",
  262k tokens, ±0.015 absolute noise; defined in §5.5), so they are noisy as well as
  biased.
- *(Interpretation:)* so every interpolated base value is *expected* to be
  pessimistic for base. This is typical behaviour, not guaranteed. Rows where base already wins would then stay
  valid.
- Matching training positions also changes base's optimizer steps. An annealed base
  run at 293M tokens takes ~8,940 steps at batch 64, against the macro models'
  17,896 at batch 32. Any E0 comparison carries this step-count confound in the macro models'
  favour.
- The equal-positions row could flip: "symmetric beats base by 0.031" and
  "Fraise ≈ base" are **not established**. Experiment E0 (§9) measures it directly.

Facts about the cost structure:
- At 100M dense parameters, the 65,536×768 output head costs about half as much as
  the transformer per position (6 × 50.3M ≈ 302 MFLOP vs ~628 MFLOP).
- Fraise's local decoder (2 blocks of local_dim 768 with SwiGLU 2048, plus the
  8,193-way head, on ~2.7 slots per position) costs a similar amount, ≈ 258 MFLOP
  per position.
- Neither cost per position shrinks with macro compression. *(Interpretation:)* so
  they erase much of the saving from
  41% fewer positions.
- The output-head cost grows with d, not with depth. The local decoder's cost grows
  with `local_dim` and `local_num_layers`. In the pilot `local_dim` was set equal to
  the transformer's width (768) and `local_num_layers` to 2, untuned (§A.2). How these shares change at larger scale is discussed
  in §7.3 (opinion).
- All numbers here are for a 512-position context. At longer contexts, attention
  FLOPs and KV-cache size grow with the number of positions, so the 0.59× position
  ratio would account for a larger share of total cost. Longer contexts were not
  tested.
- Approximate inference (forward-only) FLOPs per base token: base ~222M, symmetric
  ~182M, Fraise ~174M (of which ~50M is local-decoder calls).
- At inference, Fraise also makes sequential *local* decoder calls: one per slot, on
  average 2.705 per macro and at most 9 (L + 1). That is ≈1.59 per base token
  including END decisions, fewer when a trie node has a single option and the call is
  skipped. Measured decode speed is in §5.5g.

### 5.5 Post-hoc analysis of the pilot checkpoints (no training)

Scripts are in `data/experiments/poc-fineweb/`: `analyze.py`, `chain.py`,
`analyze_common.py` (training-time validation slice) and `analyze2.py` (full held-out
reserve). Logs are `logs/analysis.log`, `logs/chain.log` and `logs/analysis2.log`.
All re-scoring uses the final checkpoints in FP32 on GPU.

- **Slice A:** the training-time validation windows (262,144 base tokens). Re-scoring
  reproduces the logged BPB to 4 decimals (1.1302 / 1.1940 / 1.2332). Symmetric and
  Fraise have identical windows and targets there (302 windows, 154,293 macros).
- **Slice B:** the **entire 0.05 GB held-out reserve**: 13,523,544 base tokens,
  50,005,479 bytes, every model scored on identical base tokens.
  - Each model packs the stream into non-overlapping 512-position windows exactly as
    in training (stride 512, no context carried across windows).
  - Context is therefore 512 base tokens per window for base and ~873 for the macro
    models. The first target of each window has one position of context.
  - The per-base-token NLLs of all three models are aligned by position in the
    shared base-token stream (the macro stream expands exactly to the base stream).
- **Validation of the scripts:**
  - re-scoring reproduces the training-time BPB to 4 decimals;
  - the chain decomposition's per-slot sums reproduce each model's per-macro totals
    (the per-slot sums reproduce each model's per-macro NLL totals, and their
    difference, +26,401 nats = 0.171 × 154,293, matches the per-macro gap);
  - the boundary/inside split reproduces each model's total deficit to base from its
    BPB (§5.5e "Check" column).
  - The scripts have no unit tests.
- Slice B supersedes Slice A for absolute numbers.

**(a) Absolute BPB and uncertainty (Slice B).** Bootstrap by resampling 4,096-token
blocks of the stream (2,000 resamples). Blocks are much larger than a 512-position
window, so the intervals account for local correlation better than a per-window
bootstrap (the document length distribution was not checked). Seed and data-order variance remain **unmeasured**.

| Model | BPB (full reserve) | 95% CI |
|---|---:|---|
| Base 8k | 1.1356 | [1.1318, 1.1395] |
| Symmetric 64k | 1.1989 | [1.1946, 1.2033] |
| Fraise | 1.2380 | [1.2338, 1.2420] |
| Fraise − symmetric (paired) | +0.0391 | [+0.0386, +0.0395] |

On Slice A the same quantities were 1.1302 / 1.1940 / 1.2332. The window bootstrap
there gave a paired gap of +0.0392 [+0.0369, +0.0414] and ±0.015 on absolute values.
Rankings and gaps are the same on both slices, but absolute values from the 262k
slice carry ~±0.015 of sampling noise.

**(b) Trie masking (Slice A).**
- Fraise puts on average **99.86%** of each local softmax's probability on valid
  continuations (trie children, plus END where allowed).
- Renormalizing over valid continuations improves BPB by only 0.0008 (1.2332 →
  1.2324).
- Trie-masked *training* was not tested. *(Interpretation:)* it is unlikely to matter
  much, because only ~0.14% of mass is invalid.

**(c) Fraise vs symmetric by target macro (Slice A).** The gap is +0.171 nats per
target macro, +26,401 nats in total.

*By target-macro length:*

| Length | Share of targets | Symmetric NLL | Fraise NLL | Δ | Share of gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 47.0% | 4.434 | 4.633 | +0.199 | 54.6% |
| 2 | 39.5% | 5.660 | 5.819 | +0.159 | 36.6% |
| 3 | 10.8% | 6.539 | 6.658 | +0.119 | 7.5% |
| 4 | 2.1% | 6.849 | 6.924 | +0.075 | 0.9% |
| 5–8 | 0.5% | — | — | +0.05 to +0.22 | 0.4% |

*By how often the target macro occurs in the pilot's training split.* All
293,203,738 training macros were counted: 31.9% of the vocabulary occurs fewer than
1,000 times, 0.7% fewer than 100, and 60 macros never.

| Training occurrences | Share of targets | Symmetric | Fraise | Δ | Share of gap |
|---|---:|---:|---:|---:|---:|
| < 100 | ~0.0% | 17.11 | 14.30 | −2.81 | ~0% |
| 100 – 1k | 5.2% | 7.725 | 7.416 | **−0.309** | −9.3% |
| 1k – 10k | 36.8% | 6.102 | 6.319 | +0.218 | 46.8% |
| 10k – 100k | 40.6% | 4.839 | 5.038 | +0.200 | 47.3% |
| ≥ 100k | 17.5% | 3.446 | 3.595 | +0.149 | 15.2% |

Fraise is better on rare macros and worse on frequent ones. Its per-macro deficit
does **not** grow with frequency; it is slightly smaller for the most frequent macros.
- Frequency is correlated with macro length; this table does not control for it.
- No per-bucket confidence intervals were computed for §5.5c–f. The rare-macro bucket
  (5.2% of targets, ~8k target occurrences) is the least precise.

**(d) Per-decision decomposition (Slice A, chain rule).**
- Symmetric's macro softmax p(m) implies the same chain Fraise models.
  - P(first base token = b) = Σ p(m) over macros starting with b.
  - Each next token and END follow as ratios of trie-subtree masses.
- This gives, for every target macro, symmetric's NLL split into the same per-slot
  decisions as Fraise's.
- *Token decisions* at slot j exist for macros longer than j. *END decisions* at slot
  j exist for macros of length exactly j. Mean NLLs are per decision of that kind.

| Slot | Token decisions | Sym token NLL | Fraise token NLL | Δ tokens (nats) | END decisions | Sym END | Fraise END | Δ END (nats) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 154,293 | 3.706 | 3.876 | **+26,110** | 0 | — | — | 0 |
| 1 | 81,743 | 2.053 | 2.064 | +932 | 72,550 | 0.388 | 0.396 | +559 |
| 2 | 20,803 | 1.163 | 1.099 | −1,320 | 60,940 | 0.130 | 0.135 | +305 |
| 3 | 4,124 | 0.530 | 0.489 | −169 | 16,679 | 0.067 | 0.067 | +11 |
| 4–8 | 1,182 | — | — | −42 | 4,124 | — | — | +15 |
| **Total** | | | | **+25,510** | | | | **+890** |

- **Net:** slot 0 accounts for +26,110 of the net +26,401 nats (99%).
- **Gross:** the positive contributions sum to ~27,930 nats; slot 0 is ~93% of them,
  slot-1 tokens 3%, END decisions 3%.
- Fraise is **better** than symmetric on tokens at slots ≥ 2 (third base token of a
  macro onward). It is slightly worse at slot 1 (second token: +0.011 nats per
  decision) and on END (0 to +0.008 per decision).

**(e) All three models on identical base tokens, by position within the canonical
macro (Slice B).**
- Base 8k makes one global prediction per base token.
- Symmetric's per-token values are its chain-decomposed decisions; Fraise's are its
  local decoder's.
- END decisions have no base-8k counterpart: symmetric 0.243, Fraise 0.248 nats per
  macro.

| Slot | Base tokens | Base 8k | Symmetric (chain) | Fraise |
|---:|---:|---:|---:|---:|
| 0 (macro start) | 7,938,721 | 3.808 | **3.730** | 3.899 |
| 1 | 4,212,000 | **1.883** | 2.051 | 2.061 |
| 2 | 1,090,285 | **0.993** | 1.148 | 1.090 |
| 3 | 218,869 | **0.443** | 0.511 | 0.467 |
| 4 | 46,683 | 0.305 | 0.323 | **0.296** |
| 5–7 | 16,985 | ~0.18 | ~0.20 | ~0.18 |

**This table omits END decisions, which only the macro models make.** Base 8k never
decides where a macro ends. To compare total cost fairly, each model's total deficit
to base can be split into the macro *boundary* (first token + END) and the *inside*
(slots ≥ 1). All values are in millions of nats on Slice B.

| | Slot 0 vs base | END (base: 0) | Boundary total | Slots ≥ 1 vs base (slots 4–7 ≈ 0) | Total vs base | Check: ΔBPB × bytes × ln 2 |
|---|---:|---:|---:|---:|---:|---:|
| Fraise | +0.72 | +1.97 | **+2.69 (76%)** | +0.86 (24%) | +3.55 | 0.1024 → 3.55 |
| Symmetric | −0.62 | +1.93 | **+1.31 (60%)** | +0.89 (40%) | +2.19 | 0.0633 → 2.19 |

Facts read from these tables:
- Per macro start *without* END, symmetric scores best (3.730), then base (3.808),
  then Fraise (3.899). *With* END added, base is best at boundaries too:
  symmetric 3.973 and Fraise 4.147, vs base 3.808.
- Inside macros (slots 1–3), base scores best, and Fraise beats symmetric from slot
  2 on.
- For both macro models the largest single item in the deficit to base is END.
- *(Interpretation:)* the slot-0 comparison with base is not like-for-like.
  - The macro models predict slot 0 *after* the canonical boundary has been decided,
    and under BPE knowing where a macro ended carries information about the next
    base token. END pays for that information; base never pays for it and never
    receives it.
  - END is therefore partly next-token information, not pure segmentation cost.
  - Fraise's slot 0 has this extra information and still scores worse than base
    (3.899 vs 3.808).
  - Slot-0 comparisons *between the two macro models* are like-for-like.
- The "boundary" / "inside" columns split by decision type, not by cause.
  - Inside-macro token decisions also pay for probability on END: 5.17M of the 5.58M
    inside decisions have END as a consistent alternative.
  - Part of the "inside" column is therefore boundary-type cost. How much was not
    estimated.

**(f) Slot-0 deficit by context (Slice B).**

*By symmetric's first-token entropy* (interpretation as a proxy is the author's): the
entropy of symmetric's marginal
distribution over the first base token, i.e. over the 8,192 depth-1 trie-node masses.
It serves as a proxy for how multi-modal the next-token distribution is. Two
cautions:
- Binning by one model's own entropy favours that model in the lowest bins
  (regression to the mean), which affects the "vs symmetric" columns.
- Per-context entropy is not the quantity a softmax bottleneck constrains, which is
  the rank of log-probabilities across contexts.

Quintiles:

| Quintile | Entropy (nats) | Symmetric | Fraise | Δ Fraise − sym | Base 8k | Δ base − sym |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.00–2.35 | 1.146 | 1.308 | +0.161 | 1.346 | +0.200 |
| 2 | 2.35–3.66 | 2.843 | 3.019 | +0.176 | 2.908 | +0.065 |
| 3 | 3.66–4.69 | 3.894 | 4.082 | +0.188 | 3.931 | +0.037 |
| 4 | 4.69–5.62 | 4.830 | 5.010 | +0.180 | 4.863 | +0.033 |
| 5 | 5.62–8.04 | 5.936 | 6.076 | +0.140 | 5.993 | +0.057 |

Fraise's deficit varies little across these quintiles in absolute terms (+0.14 to
+0.19 nats). In relative terms it varies about 6×: 14% of symmetric's NLL in Q1
against 2.4% in Q5. A roughly constant additive offset is what miscalibration (a
wrong temperature or bias at slot 0) would produce, which D6 tests. Base
8k's deficit against symmetric is largest in the lowest-entropy quintile, partly the
binning artifact noted above.

*By type of the first base token* (categories assigned in this order: word-initial =
space then letter; starts with a letter; contains a digit; contains a newline;
otherwise punctuation/other):

| Type | Share of macro starts | Symmetric | Fraise | Δ | Base 8k | Share of slot-0 gap |
|---|---:|---:|---:|---:|---:|---:|
| Word-initial (space + letter) | 64.4% | 4.500 | 4.706 | +0.206 (1.2× average) | 4.602 | 78.2% |
| Letter, no leading space | 17.8% | 2.556 | 2.691 | +0.135 | 2.603 | 14.2% |
| Punctuation/other | 14.9% | 1.912 | 1.981 | +0.069 | 1.933 | 6.1% |
| Contains digit | 2.4% | 3.434 | 3.512 | +0.078 | 3.479 | 1.1% |
| Newline | 0.5% | 2.015 | 2.149 | +0.133 | 2.055 | 0.4% |

**(g) Generation** (greedy, 128 base tokens, batch 1, 4 prompts, CUDA).

| Model | Base tokens/s | Global steps per generated base token |
|---|---:|---:|
| Base 8k | 211 | 1.00 |
| Symmetric 64k | 200 | ≈0.59 |
| Fraise | **266** | 0.603 (measured) |

Fraise also makes up to ≈1.59 local decoder calls per base token (one per slot,
including END decisions), skipped when a trie node has a single option.
- These are batch-1 timings through the streaming API, which re-decodes the full text
  after every token.
- They are **not explained**: symmetric needs ~0.59× the global steps of base yet ran
  slightly slower, and Fraise ran fastest despite its extra local calls.
- They should not be used to predict batched or large-model inference. KV-cache memory
  and batched throughput were not measured.

**(h) Non-canonical generation (Fraise only measured).**
- With sampling (temperature 1, top-p 0.95, 20 samples of 128 tokens), 19 of 20
  generations fed at least one *non-canonical* macro segmentation to the global
  transformer.
- Overall, Fraise fed 1,578 macros where canonical re-encoding of the same text gives
  1,546 (+2.1%), mostly by ending a macro early.
- Symmetric can also sample non-canonical macro sequences (e.g. `' straw'` then
  `'berry'`); its rate was not measured.
- Neither model sees such sequences in training. The effect on generation quality is
  unmeasured.

**(i) Segmentation ambiguity at decision points (Slice B, both macro models).**
Script: `analyze3.py`; log: `logs/analysis3.log`.

At a decision point, a *consistent alternative* is a choice other than the canonical
one that still spells the true text:
- the canonical choice is END, but continuing with the true next base token is also a
  valid trie path (8.7% of END decisions);
- the canonical choice is a token, but the prefix so far is already a complete macro,
  so END is consistent and the next macro would start with that token. This applies
  to 5.17M of the 13.52M token decisions, all of them at slot ≥ 1 (END is never
  valid at the root). Every base
  token is itself a length-1 macro, so after any first token END is always a
  consistent alternative. Most of the 5.17M are slot-1 decisions.

| | Symmetric | Fraise |
|---|---:|---:|
| END decisions' share of BPB | 0.0556 | 0.0568 |
| Mean mass on the consistent "continue" alternative, over all END decisions (8.7% have one) | 1.38% | 1.45% |
| Mean mass on END, averaged over all 13.52M token decisions (0 where END is not consistent) | 9.2% | 9.5% |
| Same, averaged over only the 5.17M decisions where END is consistent | ≈24% | ≈25% |
| BPB if each consistent alternative's one-step mass were credited | 1.0098 (−0.189) | 1.0561 (−0.182) |

- Canonical scoring counts probability placed on non-canonical segmentations of the
  same text as loss. A model scored on the text itself, by summing over all
  segmentations, would score better by an unknown amount. The one-step-credit figure
  indicates the plausible magnitude.
- That figure is a **heuristic magnitude, neither a proven upper nor lower bound**: it assumes each
  alternative path then spells the rest of the text with certainty, and it ignores
  how alternative paths interact. The true gain was not estimated (E12).
- *(Interpretation:)* a large share of the END mass at token decisions is legitimate:
  it covers texts
  that do not continue the macro (e.g. END after `' straw'` when the text is not
  "strawberry"). Only the share of mass on paths that do spell the true text is a
  real segmentation cost.
- Base 8k has no macro-segmentation choices. It can in principle also place mass on
  non-canonical *base*-BPE spellings of the same text, which canonical scoring
  likewise charges; that effect was not measured *(Interpretation:)* and is
  presumably smaller.
- Fraise puts slightly more mass on consistent END than symmetric (9.5% vs 9.2%)
  but gets slightly less credit (−0.182 vs −0.189). The credit, −log(p_c + p_alt)
  − (−log p_c), depends on the canonical probability p_c as well as on p_alt.
- Under the heuristic credit both macro models would fall below base's 1.1356, and
  the Fraise−symmetric gap would widen (0.039 → 0.046). The heuristic overstates the
  gain by an unknown factor, so neither implication can be relied on.

### 5.6 Earlier BabyLM runs (superseded)

- 8-layer, 45M-total models were trained on 6–30M BabyLM tokens, validated on
  FineWeb-Edu, with a BabyLM-trained macro tokenizer (1.31× compression).
- Results were 2.59–2.76 BPB, and base 8k BPB did not improve from 6M to 30M tokens.
  *(Interpretation:)* domain mismatch between BabyLM training text and FineWeb
  validation dominated, so these runs say little about architecture.
- BabyLM has been removed as a source.

---

## 6. Status of the implementation (what exists, what does not)

**Exists and tested** (195 unit tests pass on `d0fc54e`)
- Macro tokenizer.
- Base-token budgets and exact resume.
- BPB validation.
- `fraise` architecture, including:
  - the compositional/independent switch and residual ablation;
  - trie generation and prompt healing;
  - hierarchical scoring and benchmark;
  - web UI support (architecture selector, BPB display).

**Does not exist yet**
- Multi-token prediction (stage 7).
- Dynamic decomposition (stage 8).
- Hierarchical KV caches (stage 9).
- Token-internal evaluation suites (spelling, character counting, CUTE/CharBench,
  unseen compositions). One of the four tradeoff axes in §1 (access to fine-grained
  token structure) is therefore unmeasured.
- Any variant of the local decoder other than the gated-sum MLP described in §3.4.
  No attention, no mixture-of-softmaxes head.
- Training-time GPU validation (post-hoc GPU re-scoring scripts exist, §5.5).
- A sweep of the local decoder's size (`local_dim`, `local_num_layers`); the pilot's
  values were untuned.
- Generation-quality evaluation (human or model-judged), and downstream task
  evaluation. BPB is the only quality metric so far.
- Retained intermediate checkpoints: with `checkpoint_keep_last: 2` only the last two
  pilot checkpoints of each run survive (base: steps 14,000 and 15,259; macro models:
  16,000 and 17,896). The §5.2 curves cannot be re-analyzed per slot without
  retraining.
- Seed-variance estimates.
- Equal-budget annealed baselines at reduced token counts (needed for §5.4's
  equal-position axis).

**Known engineering limits**
- *Download cache key.*
  - The raw download cache key includes the exact `max_train_tokens`, so every new
    budget re-downloads the same prefix.
  - The FineWeb-Edu cap of 10.05 GB limits budgets to ~2.68B base tokens (measured:
    the failed 4B preparation reported a capacity of 2,677,161,520).
  - A 4B-token preparation failed for this reason (10 GB downloaded to
    `data/experiments/poc-fineweb/data-4b/`).
  - Raising the cap changes the cache keys of existing caches.
- *Download lock.* It is held for the entire download, so a concurrent training start
  on the same source blocks (observed once; the GPU sat idle for ~3 min).
- *CPU validation.* About 190–250 s per checkpoint for 100M models. It runs alongside
  training but leaves the GPU idle at the end of each run.
- *NPU validation* (Intel neural-processing-unit validation via OpenVINO, an
  existing repository feature) rejects hierarchical models.
- *Fraise training tail.* One host sync per minibatch, plus an uncompiled tail.
  Fraise's speed and memory relative to its FLOPs are not understood (§5.3).

---

## 7. Hypotheses and interpretation — **OPINION**

Everything in this section is the author's interpretation of §5. Labels:
- *(consistent)*: fits every relevant measurement in §5, with plausible alternatives
  remaining.
- *(weakened)*: a measurement here points against it, but does not rule it out.
- *(ruled out)*: a measurement here contradicts it directly.
- *(excluded by construction)*: impossible given the architecture as specified; no
  measurement needed.
- *(open)*: evidence for and against it is weak or mixed.
- *(speculative)*: untested.

### 7.1 Why the macro models trail, and why Fraise trails symmetric

The measured facts this section builds on (§5.5):
- (i) Against symmetric, Fraise's deficit is ~99% net (≈93% gross) at slot 0, the
  first base token of the next macro.
- (ii) That deficit varies little with symmetric's estimated first-token entropy.
  This is weak evidence, for the reasons given in §5.5f.
- (iii) It is somewhat larger per decision on word-initial tokens (1.2×).
- (iv) It is not larger for frequent macros. Fraise appears ahead on rare macros (one
  bucket of ~8k targets, no CI, not controlled for length).
- (v) With END included, base 8k beats both macro models at macro boundaries and
  inside macros at slots 1–3 (Fraise is marginally better at slot 4). Boundaries carry 76% (Fraise) and 60% (symmetric) of the deficit
  to base.
- (vi) Fraise matches or beats symmetric on continuation tokens from slot 2 on.
- (vii) At the token decisions where END would also be consistent (5.17M of 13.52M),
  both macro models put ≈24–25% of probability on END (9.2–9.5% averaged over all
  token decisions). An unmeasured part of that mass is genuinely spent on
  alternative segmentations of the same text (§5.5i).

**A. Shared cost of macro tokenization.** Applies to both macro models against base.
- *(open)* Part of the gap to base may be a scoring and training artifact:
  canonical-only likelihood charges the macro models for probability on
  non-canonical segmentations (vii).
- That END is the largest item of their deficit to base is **not** evidence for this.
  END also carries next-token information that base never pays for (§5.5e).
- How large this part is, is unknown. The 0.18–0.19 BPB one-step-credit figure of
  §5.5i is a heuristic magnitude, not a bound.
- This matters for the PoC's evaluation as much as for the architecture. Comparing
  text likelihoods would need scoring that sums over segmentations, or models that
  never put mass on non-canonical segmentations.

**B. Fraise-specific slot-0 deficit.** Candidate explanations:

1. *(excluded by construction; no measurement needed)* **Per-prefix output
   capacity.**
   - Slot 0 is always at the trie root, so a learned vector per trie node (per
     macro prefix) added to the local state cannot change slot-0 predictions.
   - Per-*macro* output capacity in the broader sense, symmetric's 65,536 output
     rows whose sum forms its slot-0 marginal, is **not** excluded.
2. *(open; evidence against it is weak)* **Softmax bottleneck of one 8k softmax at
   slot 0.**
   - For: symmetric's slot-0 marginal is a sum over 64k softmax terms, a strictly
     more expressive family than one 8,193-way softmax.
   - Against: base 8k also uses one 8k softmax and beats Fraise at slot 0, even
     though Fraise's slot 0 additionally knows that a macro boundary was decided
     (§5.5e). Their states and inputs differ, though.
   - Against, weakly: the deficit's flat entropy profile (ii).
3. *(open)* **Less informative global states h_t.**
   - Symmetric's h_t is trained by 65,536 output rows; Fraise's only through the
     local decoder.
   - Tension: slots ≥ 1 read the same h_t, and Fraise is *not* worse there (vi).
   - Possible resolution: later slots also see the emitted prefix, which carries
     most of the needed information, so they depend less on h_t than slot 0 does.
4. *(open)* **Slot-0 read-out path.**
   - Every slot shares LN(h_t) → W_c → 2 blocks → head; only slot 0 relies on h_t
     alone.
   - Slot 0 is not under-weighted in training. Per macro there is 1 slot-0 decision,
     0.705 continuation decisions and 1 END decision, and slot 0 carries ≈72% of
     Fraise's per-macro loss (3.899 of ≈5.41 nats).
   - What remains of this explanation is the *form* of the read-out: LayerNorm then a
     768→768 projection then 2 MLP blocks, versus a single linear head on the raw
     residual stream in the baselines. Whether that form limits slot 0 is untested;
     D1(a/b) tests it directly.
5. *(open)* **Input side.**
   - Fraise's inputs are composed from base embeddings plus a 32-d residual, while
     symmetric's are independent per macro. (Fraise's extra LayerNorm on h_t is on
     the output side; see B4.)
   - A weaker input representation of frequent macros would degrade h_t and hence
     slot 0. Random, barely-trained residuals of rare macros may add input noise
     (§3.4).
   - Note: the rare-macro advantage (iv) is measured by **target** frequency. It
     plausibly reflects the output side, spelling rare targets through the shared 8k
     head rather than sparsely trained output rows, and is **not** evidence about
     composed inputs. Whether composition helps or hurts inputs is unmeasured.
   - E4 tests this; E4(c) needs code.
6. *(speculative)* **Optimization.**
   - The macro models' AdamW LR was at the grid edge and not re-tuned for batch 32 or
     100M parameters. Muon's LR was never swept.
   - Total weight decay differs between the runs (§3.4).
   - Clipping frequency differs (symmetric clipped on most steps, Fraise almost
     never). Its effect is untested; the reweighting view in §5.2 is an
     interpretation.

7. *(open)* **Calibration.**
   - The slot-0 logits may simply have the wrong temperature or bias. A roughly
     constant additive deficit across entropy (§5.5f) fits this.
   - D6, a post-hoc temperature and bias refit on frozen weights, tests it in
     minutes.
   - If it recovers much of the deficit, the cause is an optimization or
     regularization artifact (B6) rather than capacity.

D6, D1–D3, D5 and E6 (§9) are the cheapest way to separate B2–B4, B6 and B7. D5 gives a
post-hoc hint and E4 the definitive test for B5.

**Why Fraise leads early** *(speculative)*:
- Shared base-level parameters (input and output) receive dense gradient from every
  macro, so Fraise learns quickly.
- The rare-*target* advantage (iv) may be the lasting output-side remnant of this,
  though it is confounded with macro length.

**What this means for later PoC stages** *(speculative)*:
- Base 8k's advantage is largest at boundary-type decisions (slot 0 and END, plus
  part of the inside cost, §5.5e).
- Stage 8 (dynamic decomposition) would give some spans per-base-token global steps,
  which removes both the boundary *and* inside decisions for those spans.
- Its value therefore depends on how well spans are selected, which `PoC.md` leaves
  open. The decomposition by decision type does not by itself argue for or against
  stage 8.
- Reducing the boundary cost needs either better slot-0 prediction (B) or less
  probability spent on segmentation ambiguity (A).

### 7.2 Candidate fixes *(speculative; ordered by expected value per cost)*

Code status: none of these is implemented. Each needs new code (~1–3 h) unless noted.

1. **LR and schedule re-tune for the macro models** (B6).
   - AdamW above 3e-3 at batch 32, a Muon LR sweep, optionally a separate LR for the
     local head and E′ (needs new optimizer groups).
   - Cheapest in code, though ~6 h of GPU (E6). It should precede architectural
     conclusions.
2. **Auxiliary training-only macro head on h_t** (B3, partly B2).
   - A 65,536-way (or top-16k + "other") linear head trained with weight 0.1–0.3
     beside the Fraise loss, discarded at inference.
   - Training cost at the 100M pilot size is ~302 MFLOP per position, which raises
     Fraise's training FLOPs per base token from ~520M to ~697M (+34%), above base's
     ~666M. At the 38M ablation size the relative increase is larger (≈ +50–56%, depending
     on how attention FLOPs are counted). That
     worsens the FLOP axis (§5.4) unless the head is restricted to top-k macros.
3. **Dedicated slot-0 read-out** (B4, partly B2).
   - For example a direct head from LN(h_t) (or raw h_t) added to the local logits
     at slot 0 only, or slot-specific blocks for slot 0.
   - Cost ≈ one extra 8k head pass per macro.
4. **Mixture-of-softmaxes at slot 0** (B2). K = 4–8 components; costs ≈ K head
   passes per macro.
5. **Reduce segmentation ambiguity** (A), applying to both macro models.
   - (a) Estimate the true marginal-likelihood gain first: beam or sampling over
     segmentations with real global steps on a small slice.
   - (b) Train with a segmentation-aware objective, or with occasional non-canonical
     segmentations so the model learns that boundary choices are interchangeable.
   - (c) At generation, canonicalize by re-encoding.
6. *Not recommended now:*
   - **Trie-masked loss:** masking at evaluation gains ≤0.001 BPB (§5.5b). Masked
     *training* is untested, but only ~0.14% of probability mass is invalid.
   - **Per-prefix residuals:** cannot touch slot 0.
   - **More local depth for all slots, as a fix for the gap:** it spends FLOPs on all
     slots, while the gap is ~99% at slot 0. Fraise is better at slots ≥ 2 and only
     slightly worse at slot 1 (+0.011) and END. E13 sweeps depth anyway,
     to measure the cost/quality of the local decoder and to test B4 indirectly.

### 7.3 About the base 8k comparison, and a proposal for success thresholds

- *(fact)* At 100M dense parameters and 500M tokens, base 8k has the best BPB per
  token.
- *(estimate)* Per FLOP and per second it also wins, on analytic FLOPs and
  interpolated mid-schedule values (§5.4).
- *(applying the proposal below to existing data)* The pilot **fails (Q)** at 100M:
  - at equal training FLOPs, Fraise is ≥ 0.076 BPB and symmetric ≥ 0.044 BPB worse
    than (interpolated) base, against an allowed 0.02;
  - (T) and (L) are unmeasured.
- *(speculative)* Annealed baselines would be expected to widen those margins.
- *(state of evidence)* The training-positions comparison is open (§5.4, E0).
- *(fact)* KV entries and global decode steps per byte at inference are ~0.59× for
  the macro models (§5.4). Whether that saving matters depends on the deployment
  setting, which `PoC.md` does not specify.
- *(speculative)* Scale effects:
  - The output-head share of compute falls as the transformer grows. At ~1B dense
    parameters (d=2048) a 65,536-way head is ~13% of per-position transformer
    matmul FLOPs, vs ~51% here (both excluding attention; ~48% here including it).
  - The local decoder's share falls only if `local_dim` and depth stay fixed while the
    transformer grows. In the pilot they were tied to the transformer's width.
  - The within-macro advantage of base 8k (§5.5e) is not a compute-overhead effect,
    and need not shrink with scale.
- *(proposal; working default until the owner decides)* Target setting: assume the
  PoC is a step toward larger models (≥ 1B dense) used for long-context, batched
  generation. That makes inference KV size and global steps (positions) a
  first-class axis, alongside quality at matched training FLOPs. Every criterion
  below should be re-read if the owner chooses otherwise.
- *(proposal; for the project owner to accept or replace)* Numeric go/no-go criteria
  for "favourable tradeoff" at a fixed scale:
  - **(Q) quality:** *canonical* BPB (as measured throughout this report) at equal
    analytic training FLOPs, within 0.02 of an *annealed* base-8k run, with ≥ 2 seeds per
    configuration.
    - If E12 shows a material text-level (segmentation-marginal) gain, re-evaluate
      (Q) on text-level BPB as well and report both.
    - Wall-clock is implementation-dependent (§5.3), so use it only as a secondary
      check.
  - **(K) memory:** ≥ 35% fewer KV entries per byte than base 8k at inference.
    Already met: 41%. Any 64k-macro model meets it by construction, so (K) does not
    distinguish Fraise from symmetric.
  - **(L) latency:** batched decode throughput (E11) not worse than base 8k at the
    deployment batch size. Fraise adds ≈1.59 sequential local calls per base token
    to its 0.59 global steps.
  - **(T) token-internal:** on the E8 suite, ≥ base-8k accuracy and clearly
    > symmetric (e.g. +10 points on spelling and character tasks).
  - **Decision rule:** Fraise "succeeds" at that scale if (Q), (T) and (L) hold; (K)
    holds for any 64k-macro model. If (T) holds but (Q) does not, report a
    memory/structure-access tradeoff rather than a quality win.
  - **Why (Q) uses total FLOPs rather than training positions:** at this scale the
    heads and the local decoder are a large share of compute (§5.4), and positions
    alone would hide that. If the owner considers transformer positions the relevant
    budget, for instance because heads become negligible at the intended scale,
    replace (Q) with the equal-positions comparison of E0.

### 7.4 About scaling to 4B tokens *(speculative)*

- The pilot ran at ~5 base tokens per dense parameter, against the PoC's ~40. All
  conclusions are early-training conclusions.
- The Fraise–symmetric gap widened through the end of the pilot, though concurrently
  with LR decay. Whether a 4B-token run would show a larger, equal or smaller gap is
  unknown; the trend is weak evidence (§5.2).
- Spending ~2 days of GPU on 4B before D1/D2, E0, E6 and one targeted fix would mostly
  re-measure a localized (slot 0) but not yet explained weakness.

### 7.5 Smaller observations *(speculative)*

- Symmetric's mid-run gradient norms (≈3 at step 5,000, 28% into the run) were large
  relative to its own later values. That fits noisy updates to its huge,
  sparsely-hit tables. Norms are not comparable across models (§5.2), so this is not
  a statement about Fraise.
- Non-canonical segmentations at generation time (+2.1% more macros than canonical
  for Fraise, §5.5h) are a train/inference mismatch for any macro-input model. They
  could differ between the two: Fraise's trie decoding can END after any prefix that
  is a macro, and symmetric can sample any macro sequence. Neither rate has been
  compared (§5.5h).
  Options: canonicalize by re-encoding before feeding the global model (costs a
  re-prefill), or train with segmentation noise. Not urgent until generation quality
  is evaluated.

---

## 8. Caveats on the measurements

- **Early training.** ~5 base tokens per dense parameter (PoC target ~40).
- **Variance.**
  - One seed and one data order per configuration. Provisional threshold for
    separately trained runs: 0.02 BPB (§9).
  - Sampling noise from the validation text, 95%:
    - ±0.015 absolute on the 262k training-time slice;
    - ±0.004 absolute on the full 13.5M-token reserve;
    - ±0.0005 for paired Fraise−symmetric differences on the full reserve (§5.5a).
  - Seed and data-order variance are unmeasured. Treat differences below 0.02 BPB
    between separately trained runs as unresolved until seeds are run.
- **Interpolated baselines** (§5.4) are read mid-schedule and are *expected* (not
  proven) to be pessimistic for
  base.
- **Learning rates and batch.**
  - Only the AdamW LR was swept (the Muon LR was not), at 38M dense parameters with
    batch 64 × 512, and transferred to 100M parameters unchanged.
  - The macro models moved to batch 32 (step count +17% vs base, tokens per step
    −15%).
  - The macro models' optimum may be above 3e-3 (edge of grid).
  - Whether the batch and step choice helps or hurts the macro models is not
    established.
- **Clipping.** Symmetric's updates were clipped on most steps; Fraise's essentially
  never (§5.2).
- **Sparse capacity.** Symmetric has ~79.7M more lexical (embedding + head)
  parameters than Fraise: 50.3M + 50.4M against 12.6M + 2.1M + 6.3M. Almost all of
  them are per-macro rows; Fraise has 10.1M more dense parameters. No run matches Fraise's
  lexical/head capacity to symmetric's (E1 and E4 in §9 partly address this).
- **Validation slices.**
  - Training-time validation used 262k tokens from the start of the held-out reserve
    (absolute noise ±0.015).
  - Post-hoc re-scoring of the final checkpoints on the full 13.5M-token reserve
    (§5.5a) narrows this to about ±0.004.
  - Intermediate checkpoints (the §5.2 curves) exist only on the small slice.
- **Dense parameters.** Fraise has 10% more dense parameters (109.3M vs 99.2M)
  because of its local decoder.
- **AdamW decay also differs**: compounded ≈0.93× for base (AdamW 1e-3) vs ≈0.76×
  for the macro models (3e-3) on embeddings, heads and tables.
- **Total weight decay differs** with step count: Muon matrices ≈0.22× (base) vs
  ≈0.17× (macro models) of compounded decay pull (§3.4).
- **Base tokenizer training data.** The 8k base tokenizer was trained on a mixture
  including BabyLM and Cosmopedia; the macro merges were trained on FineWeb-Edu. All
  three models share the base tokenizer, so this does not bias between them, but
  base tokenization may be less efficient on FineWeb than an in-domain one would be.
- **Canonical-only scoring** penalizes macro models for probability on alternative
  segmentations (§5.5i). Base 8k is unaffected by macro segmentation; its own
  non-canonical base-BPE mass is unmeasured.
- **Architecture asymmetry.** Fraise has a LayerNorm on h_t before its output path;
  the baselines read the un-normalized residual stream (§3.4). Unablated.
- **FLOP estimates** (§5.4) are analytic (6N rule plus causal attention). Measured
  wall-clock is given alongside.
- **Context length in bytes.** BPB scoring windows differ per tokenizer: 512 positions
  ≈ 873 base tokens for the macro models vs 512 for base. *(Interpretation:)*
  longer context generally helps; the size of this effect here is unmeasured (D4).
- **Context length.** 512 positions was inherited from the repository's defaults, not
  chosen for this study. The KV advantage motivating the PoC grows with context
  (§5.4), and longer contexts were not tested.
- **Domain.** All training and evaluation is English FineWeb-Edu, with an in-domain
  macro tokenizer. Code, multilingual and numeric-heavy text are untested, and the
  macro tokenizer's 0.3 GB of training text likely overlaps the first ~81M training
  tokens of the models. That overlap affects all macro-model runs equally.
- **Fraise vs symmetric is not cost-matched.** Fraise uses ~5% fewer training FLOPs
  per base token (520M vs 546M) and ~79.7M fewer lexical parameters. The 0.039 gap is
  at equal tokens and steps; at equal FLOPs it would be slightly smaller, by an
  unmeasured amount.
- **Speed trend.** At 38M, Fraise trains ~9% *slower* than base (146–150k vs
  162–165k tokens/s). At 100M it is ~2.5% faster. The local decoder's overhead
  relative to the transformer is shrinking with scale, but is not yet modelled
  (E9).
- **No trie masking in training or validation.** The evaluation-time effect was
  measured on Slice A only, at ≤0.001 BPB (§5.5b); the training-time effect is
  unmeasured.

---

## 9. Recommended next experiments (with cost estimates)

Definitions used below:
- **gap** = BPB(Fraise variant) − BPB(symmetric) on Slice B (full reserve). Positive
  means Fraise is worse.
- **Two kinds of comparison need different thresholds.**
  - *Same weights, different scoring* (§5.5, D3, D4): differences above 0.005 BPB
    are real: ≈11× the half-width of the model-vs-model paired CI. *Probes* (D1, D2) add training variance: count a probe effect only
    if it exceeds 0.005 BPB on Slice B *and* reproduces
    with a second probe training seed. In slot-0 units that is ≈0.022 nats per
    macro start (1 nat per macro start ≈ 0.229 BPB on Slice B).
  - *Separately trained runs* (E-experiments): seed variance is unknown, so until E5
    measures it, treat differences below **0.02 BPB** as unresolved. 0.02 is an
    assumed placeholder (about half the pilot gap), not derived from data. After E5,
    use 2× the measured seed standard deviation of the paired gap.
- A change **closes the gap** if the *reduction* in the gap is at least half the
  reference gap and also exceeds the applicable threshold, i.e. reduction ≥
  max(gap/2, 0.02). A reference gap below 0.02 cannot be resolved at all. Between
  0.02 and 0.04, closing requires a reduction of ≥ 0.02, more than half the gap.
  The pilot's 0.039 qualifies.
- **Seeds.** "≥ 2 seeds per configuration" means the existing run plus at least one
  more. Decisive comparisons should use 3.
- **Sensitivity caveat.** With the pilot's 0.039 gap, half the gap (≈0.02 BPB) equals
  the provisional separately-trained threshold. Until seeds are measured, ablations
  can detect only effects of about the whole gap. Run E5 early, or use ≥ 3 seeds for
  the decisive comparisons.
- **Evaluation:** every run is evaluated post hoc on Slice B with `analyze2.py`, so
  absolute and per-slot numbers are available, not only the 262k training-time
  slice.

| ID | Experiment | Question answered | Cost |
|---|---|---|---|
| D1 | **Probe on frozen Fraise h_t:** train (a) a linear and (b) a 2-block-MLP first-token head (8k + END) on raw h_t, and (c) a 65,536-way linear macro head marginalized to the first token. Train on the pilot's *training* split until held-out probe loss changes < 0.5% over the last 20% of steps (expect 100–300M tokens for (c), whose head has 50M parameters); evaluate slot-0 NLL on Slice B | Does h_t contain the information Fraise's slot-0 path fails to read (B4, B2), or is h_t weaker (B3)? | ~3 h incl. code |
| D3 | **Local-decoder attribution on frozen Fraise:** measure norms of W_c·LN(h_t) vs the prefix sum per slot, and the Slice-B NLL change per slot when the h_t term is zeroed for slots ≥ 1 | How much later slots rely on h_t; informs (does not settle) the B3/B4 tension in §7.1 | ~1 h |
| D1/D2 setup | Probes: AdamW (lr 1e-3, betas 0.9/0.95, no weight decay), batch 32 × 512 positions of frozen BF16 features, cosine schedule; 2 probe seeds each. D1(a): linear on raw h_t. D1(b): W_c-like projection + 2 SwiGLU blocks + head on raw h_t (no LayerNorm). D1(c): trained with macro cross-entropy, then marginalized to the first token at evaluation. D2(a): as D1(a). D2(b): Fraise's slot-0 path, including LayerNorm, trained on slot 0 only | — | — |
| D6 | **Slot-0 calibration refit** on frozen Fraise: fit a temperature and a per-token bias (8,193 values) on slot-0 logits using 10M training-split tokens; evaluate on Slice B | Is the slot-0 deficit an additive miscalibration (B7)? | ~30 min |
| D2 | **Probe on frozen symmetric h_t:** (a) linear 8k first-token head, (b) Fraise-style slot-0 path; same protocol | Can a single 8k read-out match symmetric's 64k marginal (3.730 nats) given symmetric's states? | ~2 h |
| E0 | **Annealed base 8k** at 293M, 391M and 410M tokens (pilot config, schedule ending at the budget, base LR from E6b), evaluated on Slice B; plus a **step-matched control** at 293M with batch 32 (~17.9k steps, equal to the macro runs) | Narrows the training-positions and FLOP rows of §5.4 (seed and context confounds remain) | ~5.2 h (1,387M tokens at ~81k/s plus validation) |
| E6b | **Base LR check** at the comparison scale: base AdamW {3e-4, 1e-3, 3e-3}, for use in E0 and E14 | Fair base references | ~1.5 h at 38M; ~3.3 h at 100M / 293M |
| E14b | Fraise **without** the LayerNorm on h_t before W_c (needs a config flag) | Tests the form part of B4 directly; complements E14 | ~35 min + ~0.5 h code |
| E6 | **LR re-sweep, macro models at batch 32:** AdamW {1.5e-3, 3e-3, 6e-3, 1.2e-2} × {symmetric, Fraise} (both directions: the smaller batch, larger model and longer run of the pilot may lower the optimum), then Muon {0.01, 0.02, 0.04} at the best AdamW, at the ablation scale; optionally separate local-head/E′ LR (needs code) | §7.2 fix 1 (B6); prerequisite for fair ablations | ~7 h |
| S | **Scale check:** at E6's best LRs, does Fraise trail symmetric at the ablation scale (38M dense / 300M tokens) by ≥ 0.02 BPB at the end of training, *and* is the deficit again concentrated at slot 0 (`analyze2.py`)? | Is the 38M / 300M regime representative? | (included in E6) |
| E1 | Fraise + auxiliary training-only macro head (weights 0.1, 0.3) | §7.2 fix 2 | ~50 min each (≈ +50–56% FLOPs at 38M; check memory) + ~1.5 h code |
| E2 | Fraise + direct slot-0 head from h_t | §7.2 fix 3 | ~35 min + ~1 h code |
| E3 | Fraise + MoS at slot 0 (K = 4) | §7.2 fix 4 | ~40 min + ~2 h code |
| E4 | Input-side hybrids: (a) Fraise with `macro_embedding: "independent"`, `macro_residual_dim: 0` (supported by config); (b) Fraise with `macro_residual_dim: 0`, compositional without residuals (supported); (c) symmetric with compositional inputs (needs code) | Input-side contribution; also validates stages 4–5 | ~1.8 h (+1 h code for c) |
| E5 | Seeds: repeat symmetric, Fraise (and later the best fix) with 2 more seeds at the scale being compared. With one source and `shuffle_per_epoch: false`, changing `seed` varies only initialization. To also vary data order, set `shuffle_per_epoch: true` (document order then depends on the seed) | Init (and optionally data-order) variance for all comparisons | ~2.5 h at 38M; ~10 h at 100M/500M |
| E7 | Macro vocabulary: 32k and 128k vocabularies, and `max_macro_length` 4 vs 8, for Fraise and symmetric. Compare at equal training tokens on BPB, and report positions per byte and training FLOPs per byte alongside, since both change with the vocabulary | Compression vs rarity tradeoff (§3.2.7) | ~4 h incl. tokenizers |
| E8 | Token-internal suite (proposed protocol): templated prompts scored by likelihood of the correct vs distractor completions (no sampling), so all models are compared by the same rule. Score each candidate with its canonical encoding (as for BPB) and compare *relative* accuracy across models; the segmentation-ambiguity cost (§5.5i) affects absolute log-likelihoods and could bias comparisons between candidates of different macro lengths, so report accuracy per macro-length bucket. ≥ 500 items per task, words drawn from frequency bands of the training split, 3–4 distractors differing by one character or position; significance by paired bootstrap over items. Tasks: spell a word ("s-t-r-a-w"), count a letter, n-th character, reverse a short word, add/remove a prefix or suffix, digit-level addition, and frequent vs rare vs unseen word compositions. Metric: accuracy per task; report per macro-length bucket. Run on the pilot checkpoints | The structure-access axis (§7.3 T) | ~½ day implementation |
| E9 | Profile Fraise training (time and memory) | Explains §5.3; needed for cost predictions | ~1 h |
| E10 | 4B-token runs at 100M dense: raise the FineWeb cap to ~16 GB and stop keying downloads by exact budget first | Scaling behaviour | ~2 days GPU |
| E11 | Batched inference benchmark: KV memory and tokens/s at batch 16–64 | Inference-side tradeoff (§7.3 K and L) | ~2 h |
| E12 | Segmentation-marginal estimate: on ~200k base tokens of Slice B, beam (width 8) over segmentations with real global steps, summing path probabilities, for both macro models | How much of §5.5i's ambiguity cost is real (replaces the 0.19 BPB heuristic magnitude with an estimate) | ~3 h incl. code |
| E14 | LayerNorm asymmetry: base and symmetric with a final LayerNorm before `lm_head` (needs a config flag), each against a same-scale reference run without it (base needs its own 38M / 300M reference; E6 covers only the macro models) | Removes the §8 architecture confound | ~2.2 h + ~0.5 h code |
| D5 | More post-hoc breakdowns on existing weights: slot-0 deficit by the *input* (previous) macro's training frequency and length; by position within the window; and the per-slot comparison on the sweep's final 38M / 50M weights (`sweep-*-0.003`), where Fraise led | Gives a post-hoc hint about B5; separates context-depth effects; shows whether Fraise's lead in a *different, smaller and shorter-trained* model was also at slot 0 (only a proxy for the pilot's early phase, whose checkpoints are gone) | ~1 h |
| D4 | Equal-context re-scoring: score the macro models with ~300-position windows (≈512 base tokens of context, like base) on Slice B | Removes the context-length bias from §5.5e comparisons | ~30 min |
| E13 | Local decoder size at 38M dense (d = 512): `local_dim` {256, 512} × `local_num_layers` {1, 2, 4}, against the same-scale symmetric reference | Cost/quality of the local decoder; needed for FLOP-axis projections | ~3.5 h |

Settings for E1–E7 unless stated (E6's tuned LRs are assumed to transfer from 38M to
100M; whether they do is itself unverified, so re-check the best setting at 100M
before the pilot-scale confirmation):
- 38M-dense configs of §A.4, 300M base tokens.
- Schedule: warmup 200, cosine to zero; Muon 0.02.
- AdamW at E6's best value for each model.
- Macro models at batch 32 × 512, base at 64 × 512, as in the pilot.
- If S shows no gap or a different mechanism at 38M / 300M, run the ablations at the
  pilot scale (100M dense / 500M tokens, ~1.7 h each), reusing the pilot runs as
  references. At ~300M tokens the pilot's mid-schedule gap was ~0.02. That is only a
  weak guide, since an annealed 300M-token run would differ, but it suggests a
  300M-token ablation at 100M may be too small to resolve.
- The 38M config has attention width 768 with d = 512, unlike the 100M config
  (width = d). Conclusions at 38M should be confirmed at 100M before acting on them.
- E0 and base comparisons should use a base AdamW LR re-checked at the comparison
  scale. Base's 1e-3 also comes from the 50M-token sweep and was not re-tuned.
- Throughputs at 38M dense (§4) put 300M tokens at ~27–34 min per run.

Decision guide.
- **"Recovers"** means closing at least half of Fraise's slot-0 deficit of 0.169
  nats per macro start.
- **"Recovers nothing"** means closing less than 0.022 nats.
- **"Partial"** is in between, judged on the best-performing probe.
- **D6 recovers** (calibration alone closes ≥ half the deficit): treat the gap as an
  optimization or regularization artifact. Run E6 before any architectural fix.
- **D1(a) recovers** (a plain linear 8k head on raw h_t): h_t has the information
  and a simpler read-out than Fraise's suffices. Either the form of Fraise's read-out
  (B4) or its optimization within joint multi-slot training (B6) is implicated, and
  B2 is weakened. Run E6 first if not yet done, then E2 (fix 3).
- **D1(b) recovers but D1(a) does not:** the probe has nearly the same form as
  Fraise's own slot-0 path, except that it reads raw h_t without Fraise's LayerNorm.
  Form (apart from that LayerNorm, which E14b tests) is then *not* the cause. Likely causes are slot
  sharing or joint training of the path with continuations and END. Try
  slot-0-specific blocks (a variant of fix 3) before MoS.
- **D1(c) and at least one of D1(a)/(b) recover:** apply the D1(a) or D1(b) branch
  above, and prefer the cheapest fix.
- **Partial recovery (more than 0.022 but less than 0.085 nats per macro start):**
  several causes contribute. Run E2 and E6 first, then E1.
- **Only D1(c) recovers:** a single softmax cannot express it but a macro-level
  read-out can (B2). Prioritize fix 4 (E3), and fix 2 (E1) as a training-only
  alternative.
- **D1 recovers nothing (converged probes) and D2(a) matches symmetric within
  0.022 nats:** Fraise's h_t is weaker (B3, or input-side B5). Prioritize E1 and E4.
- **D1 recovers nothing and D2(a) does not match:** two possibilities.
  - Fraise's h_t lacks the information (B3, or input-side B5), since even D1(c)
    cannot extract it.
  - A single 8k read-out is also insufficient (B2).
  - Prioritize E1 and E4 (states), then E3.
- **A probe has not converged:** do not draw conclusions from it.
- **E6 alone closes the gap:** it was optimization. Redo the pilot comparison with
  tuned LRs before any architectural change.
- **A variant closes the gap at the ablation scale:** confirm at 100M / 500M with
  ≥ 2 seeds per configuration (E5), then consider E10.
- **S finds no gap at 38M:** test B6 at 100M instead (E6 at 100M / 300M:
  8 runs × ~1.1 h ≈ 9 h).
- **Other diagnostics:**
  - D2(b) ≈ D2(a): Fraise's read-out form is adequate given good states.
  - D2(b) worse than D2(a): the form costs something even with good states (B4).
  - D3: if zeroing the h_t term barely hurts slots ≥ 1, later slots run mostly on
    the prefix. That resolves the B3/B4 tension in favour of "h_t matters mainly at
    slot 0".
  - D4: if equal-context scoring shrinks the macro models' per-slot advantage over
    base, re-read §5.5e with that correction.
  - D5: if the slot-0 deficit concentrates after rare *input* macros, B5 (input
    side) gains support; then run E4 first.
  - E4: a gap change ≥ threshold attributes that much to inputs; otherwise drop B5.
  - E7, E13: choose by BPB at equal training FLOPs. Report positions per byte
    alongside.
  - E14 / E14b: if adding LayerNorm changes a baseline, or removing it changes
    Fraise, by ≥ threshold, correct all cross-model comparisons for it.
- **None closes the gap:** report Fraise as currently less sample-efficient than
  symmetric on BPB at this scale. Its case then rests on E8, E12 and the inference
  memory axis. Failed fixes do not establish a fundamental limit.
- **E0 at ~293M tokens** (annealed base 8k on Slice B; compare with symmetric 1.199
  and Fraise 1.238; seed variance applies, as these are separate runs):
  - Compare both macro models with the 293M run (positions), symmetric with the 410M
    run (its FLOP point) and Fraise with the 391M run (its FLOP point).
  - Base ≥ 0.02 worse than a macro model: that model provisionally wins on that
    axis. Confirm with D4-style equal-context scoring and seeds, because the step-count
    and context-length confounds both favour the macro models.
  - Within ±0.02: unresolved until seeds are run (E5 extended to base).
  - Base ≥ 0.02 better than both: neither macro model wins on any **BPB** axis at
    100M scale. The evaluation should then rest on E8, E12, inference memory and
    larger-scale projections.
  - Remember the step-count confound in the macro models' favour (§5.4); the
    total-decay difference (§3.4) goes with it.

All thresholds, orderings and decision rules in this section are the author's
recommendations.

Ordering against PoC stages 7–9 *(opinion)*:
- Finish D6, D1–D5, E6b, E0, E6 and S first (about 1.5 days of GPU). The table above
  is grouped by kind (diagnostics, references, fixes, studies), not by priority;
  this list gives the order.
- Then either one slot-0 fix (if a diagnostic points at one) or stage 8 (if E8 shows
  Fraise's structure-access advantage and the slot-0 gap persists). Stage 8 changes
  the comparison itself, so its prototype should be evaluated with the same
  `analyze2.py` breakdown.
- Stage 7 (MTP) is a decoding-speed optimization and can wait until the quality
  picture is settled.

---

## 10. Reproduction

Post-hoc analysis scripts are **not in the repository commit**. They live in
`data/experiments/poc-fineweb/` (`analyze.py`, `analyze_common.py`, `chain.py`,
`analyze2.py`, `analyze3.py`) and hard-code that directory and the pilot run
names. Run them from a checkout with `uv run python <script>`; each takes 5–30 min
on the GPU.

From a checkout at `d0fc54e`:

```sh
uv sync --extra web
E=data/experiments/poc-fineweb       # contains configs/ used below

# Macro tokenizer (≈4 min)
uv run create-tokenizer --data-dir $E --base-tokenizer $E/tokenizers/base-8192.json \
  --vocab-size 65536 --max-macro-length 8 --sources fineweb-edu --source-size-gb 0.3

# Pilot run (example: Fraise; ≈1.7 h on an RTX 5090 Laptop)
uv run train --data-dir $E --training-hyperparameters $E/configs/pilot-macro.json \
  --model-config $E/configs/100m/fraise.json \
  --tokenizer-file $E/tokenizers/fraise-macro-65536-b501b7e0c0e323b6.json \
  --model-name pilot-fraise --device cuda

# Parameter breakdown
uv run params $E/configs/100m/fraise.json

# Tests
uv run --extra web python -m unittest discover -s tests
```

- Validation history per run is in
  `$E/models/<run>/model_meta.json → validation_results`; logs are in
  `$E/logs/<run>.log`.
- Final weights are `$E/models/<run>/<run>.pt`.
- Load with `fraise.inference.artifacts.load_run`, then `iter_generate` /
  `score_text`.

---

## Appendix A. Exact configurations

The JSON below is copied from the archived run artifacts in
`data/experiments/poc-fineweb/models/<run>/`:
- `model_config.json` and `training_hyperparameters.json` are the exact bytes
  supplied at launch.
- `model_meta.json → manifest.identity` holds the normalized settings, with every
  default filled in.

Where a field does not appear in a supplied file, its value is the default from
`fraise.config.training.TrainingSettings` or the model config dataclass. The
normalized blocks list every field explicitly.

### A.1 Tokenizers

| Role | File (in `data/experiments/poc-fineweb/tokenizers/`) | SHA-256 of serialized tokenizer |
|---|---|---|
| Base 8k (byte-level BPE, 8,192) | `base-8192.json` (copy of `data/tokenizers/fraise-8192-babylm-cosmopedia-khanacademy-cosmopedia-stories-fineweb-edu.json`) | `7add757565786d7ce1dda17dec6a90c64f7de3e11b62753690261279e5a024d1` |
| Macro 64k (over the base above) | `fraise-macro-65536-b501b7e0c0e323b6.json` | `713945ab48ea974f38021f58dcce4dafbea8440a794a86e38ff091f9479d34db` |

Macro tokenizer creation request (`tokenizers/metadata.json`):

```json
{
  "kind": "macro",
  "base_tokenizer_sha256": "7add757565786d7ce1dda17dec6a90c64f7de3e11b62753690261279e5a024d1",
  "dataset_fingerprint": "d1ce10181500c153e875758509402d49f2101f23b9cb832f1cd5a3cc11112e6f",
  "sources": ["fineweb-edu"],
  "vocab_size": 65536,
  "max_macro_length": 8,
  "min_frequency": 2,
  "max_bytes_per_source": null
}
```

The training text was the FineWeb-Edu prefix after the 0.05 GB validation reserve,
with a 0.3 GB download cap (`--source-size-gb 0.3`).

### A.2 Pilot model configs (`configs/100m/*.json`, identical to the archived `model_config.json`)

**Base 8k/8k** (`pilot-base`)
```json
{
  "architecture": "transformer",
  "vocab_size": 8192,
  "residual_stream_dim": 768,
  "head_dim": 64,
  "qk_dim": 64,
  "num_heads": 12,
  "swiglu_up_dim": 2048,
  "num_layers": 14,
  "max_seq_len": 512,
  "tie_word_embeddings": false
}
```

**Symmetric 64k/64k** (`pilot-symmetric`): identical except `"vocab_size": 65536`.

**Fraise** (`pilot-fraise`)
```json
{
  "architecture": "fraise",
  "vocab_size": 65536,
  "base_vocab_size": 8192,
  "max_macro_length": 8,
  "residual_stream_dim": 768,
  "head_dim": 64,
  "qk_dim": 64,
  "num_heads": 12,
  "swiglu_up_dim": 2048,
  "num_layers": 14,
  "max_seq_len": 512,
  "macro_embedding": "compositional",
  "macro_residual_dim": 32,
  "local_dim": 768,
  "local_num_layers": 2,
  "local_swiglu_up_dim": 2048
}
```

### A.3 Pilot training hyperparameters

Supplied file for **symmetric and Fraise** (`configs/pilot-macro.json`, byte-identical
to both runs' `training_hyperparameters.json`):

```json
{
  "batch_size": 32,
  "minibatch_size": 32,
  "learning_rate": 0.02,
  "adamw_learning_rate": 0.003,
  "weight_decay": 0.01,
  "num_epochs": 1,
  "sequence_length": 512,
  "warmup_steps": 200,
  "max_train_tokens": 500000000,
  "dataset_token_counts": {
    "fineweb-edu": 500000000,
    "cosmopedia-khanacademy": 0,
    "cosmopedia-stories": 0
  },
  "max_validation_tokens": 262144,
  "validation_timeout_seconds": 1200,
  "validation_device": "cpu",
  "use_bf16": true,
  "use_compile": true,
  "checkpoint_interval_steps": 2000,
  "checkpoint_keep_last": 2,
  "log_interval_steps": 200,
  "shuffle_per_epoch": false,
  "shuffle_data": true
}
```

**Base 8k** (`configs/pilot-base.json`): identical except
`"batch_size": 64` and `"adamw_learning_rate": 0.001` (`minibatch_size` stays 32,
i.e. two accumulated minibatches per step).

Normalized settings for the macro pilot runs (all defaults explicit; base differs
only in `batch_size` 64 and `adamw_learning_rate` 0.001):

```json
{
  "adamw_learning_rate": 0.003,
  "batch_size": 32,
  "betas": [0.9, 0.95],
  "checkpoint_interval_steps": 2000,
  "checkpoint_keep_last": 2,
  "compile_threads": 4,
  "dataset_token_counts": {"cosmopedia-khanacademy": 0, "cosmopedia-stories": 0, "fineweb-edu": 500000000},
  "deterministic": false,
  "gradient_clip_norm": 1.0,
  "learning_rate": 0.02,
  "log_interval_steps": 200,
  "max_train_tokens": 500000000,
  "max_validation_tokens": 262144,
  "minibatch_size": 32,
  "muon_momentum": 0.95,
  "muon_nesterov": true,
  "muon_ns_steps": 5,
  "num_epochs": 1,
  "packing_stride": 512,
  "seed": 1337,
  "sequence_length": 512,
  "shuffle_data": true,
  "shuffle_per_epoch": false,
  "tied_weight_init_std": 0.02,
  "tokenization_batch_size": 1000,
  "use_bf16": true,
  "use_compile": true,
  "validation_cpu_threads": 4,
  "validation_device": "cpu",
  "validation_timeout_seconds": 1200,
  "warmup_steps": 200,
  "weight_decay": 0.01
}
```

Field meanings that matter for interpretation:
- `learning_rate` is the **Muon** learning rate for hidden 2-D matrices (cosine
  schedule with `warmup_steps`). `adamw_learning_rate` and `betas` apply to
  embeddings, output heads, residual tables and vectors; vectors get no weight decay.
- `batch_size` counts *sequences* of `sequence_length` *positions* per optimizer step.
  It is not tokens, and not base tokens for the macro models.
- `minibatch_size` only splits a step for memory. Gradients are accumulated
  token-weighted, so it does not change the update.
- `max_train_tokens` / `dataset_token_counts` count **base tokens** for every
  tokenizer (§3.3).
- `max_validation_tokens` counts base tokens of the held-out prefix scored at each
  checkpoint.
- `packing_stride` equal to `sequence_length` means non-overlapping windows.

Execution provenance (pilot runs, from `manifest.provenance`):
- Python 3.14.8, torch 2.14.0+cu130, CUDA device.
- BF16 mixed precision, `torch.compile` over forward+loss, FP32 matmul precision
  "high" (TF32), attention backend auto, non-deterministic.
- Git commit `d0fc54e`, recorded as `dirty: true`. Tracked source and test files had
  no modifications relative to that commit; the flag reflects untracked files in the
  worktree.

### A.4 LR sweep configs

Model configs (archived `model_config.json`):

```json
// sweep-base-*  (symmetric identical except "vocab_size": 65536)
{"architecture": "transformer", "vocab_size": 8192, "residual_stream_dim": 512, "head_dim": 64,
 "qk_dim": 64, "num_heads": 12, "swiglu_up_dim": 1024, "num_layers": 12, "max_seq_len": 512,
 "tie_word_embeddings": false}

// sweep-fraise-*
{"architecture": "fraise", "vocab_size": 65536, "base_vocab_size": 8192, "max_macro_length": 8,
 "residual_stream_dim": 512, "head_dim": 64, "qk_dim": 64, "num_heads": 12, "swiglu_up_dim": 1024,
 "num_layers": 12, "max_seq_len": 512, "macro_embedding": "compositional", "macro_residual_dim": 32,
 "local_dim": 512, "local_num_layers": 2, "local_swiglu_up_dim": 1024}
```

Normalized training settings shared by every sweep run (example `sweep-fraise-0.003`):

```json
{"adamw_learning_rate": 0.003, "batch_size": 64, "betas": [0.9, 0.95],
 "checkpoint_interval_steps": 100000, "checkpoint_keep_last": 2, "compile_threads": 4,
 "dataset_token_counts": {"cosmopedia-khanacademy": 0, "cosmopedia-stories": 0, "fineweb-edu": 50000000},
 "deterministic": false, "gradient_clip_norm": 1.0, "learning_rate": 0.02, "log_interval_steps": 100,
 "max_train_tokens": 50000000, "max_validation_tokens": 262144, "minibatch_size": 16,
 "muon_momentum": 0.95, "muon_nesterov": true, "muon_ns_steps": 5, "num_epochs": 1,
 "packing_stride": 512, "seed": 1337, "sequence_length": 512, "shuffle_data": true,
 "shuffle_per_epoch": false, "tied_weight_init_std": 0.02, "tokenization_batch_size": 1000,
 "use_bf16": true, "use_compile": true, "validation_cpu_threads": 4, "validation_device": "cpu",
 "validation_timeout_seconds": 900, "warmup_steps": 50, "weight_decay": 0.01}
```

Per-run differences:
- `adamw_learning_rate` takes the value in the run name (0.003, 0.001 or 0.0003).
- `minibatch_size` varied as below. It does not change the optimization (it only
  splits the 64-sequence step), so sweep results are comparable across it.

| Run | minibatch_size | Reason |
|---|---|---|
| sweep-base-0.003, sweep-symmetric-0.003 | 16 | started before the switch to 64 |
| sweep-base-0.001, sweep-base-0.0003, sweep-symmetric-0.001, sweep-symmetric-0.0003 | 64 | |
| sweep-fraise-* (all three) | 16 | out of memory at 64; automatic retry at 16 |

With only one validation at the end (`checkpoint_interval_steps` 100,000 exceeds the
run length), every sweep result is the final-step BPB.

### A.5 Benchmark configuration (§5.3)

`data/experiments/poc-fineweb/bench.py`:
- The 100M model configs of A.2 and the optimizer of the training runtime (Muon +
  AdamW), with BF16 autocast and the compiled loss.
- 64 sequences × 512 positions per step at minibatch 16/32/64.
- Real FineWeb-Edu held-out tokens.
- Three warm-up steps, then ten timed steps.

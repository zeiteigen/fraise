# tokenization-workbench

The tokenization workbench (formerly fraise-web's **Tokenization** view) as an AblationKit project
panel (Project → Tokenization workbench).

Type text and compare up to six configurations side by side. By default these are the PoC pilot's
three runs, each with the hierarchy its own run was trained with:

| Default | Tokenizer | Hierarchy shown |
| --- | --- | --- |
| `pilot-base-8k` | `bpe-8k` (8k byte-level BPE) | none (plain BPE) |
| `pilot-hier-64k` | `hierarchical-64k` (64k hierarchical tokenizer over `bpe-8k`) | none (`independent` embeddings) |
| `pilot-fraise-64-8` | `hierarchical-64k` in, `bpe-8k` out | two-level 8k → 64k, 32-d residuals |

Other configurations: a tokenizer's default two-level hierarchy, merge-prefix levels, a file in
`DATA_DIR/hierarchies/`, or segmentation only. Each card shows the segmentation strips, the
composition icicle, the level and residual table, the composition trees and a node inspector.

## How it works
- **Python** (`python/workbench.py`) runs in fraise's own environment (`python.environment:
  "project"`) at the checkout's current commit. It checks requests with
  `fraise.workbench.requests.TokenizeRequest` and calls `fraise.workbench.catalog.Catalog` and
  `fraise.workbench.tokenization.TokenizationWorkbench.analyze`, which call
  `fraise.preview.hierarchy.analyze_text`, over `DATA_DIR` = `$ABLATIONKIT_CHECKOUT/data`
  (`fraise.paths.data_root`). It reimplements nothing.
- **The view** (`ui/src/`) collects input and draws the result. The bridge carries at most 256 KiB
  of result data, so the analysis arrives packed (columnar) and gzipped by
  `fraise.workbench.payload`, losslessly. In practice that allows a few thousand characters with
  three configurations; longer text gets a clear error.
- **Capability:** `project.analysis.execute` only, to run its two Python contributions. Python has
  no network; it reads fraise's data directory as trusted code under your user.

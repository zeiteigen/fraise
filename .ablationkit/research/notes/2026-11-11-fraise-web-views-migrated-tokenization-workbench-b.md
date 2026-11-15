<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261111-544ed3","project_id":"fraise","kind":"note","tags":["onboarding","extension","visualisation"],"links":[{"kind":"report","id":"rep-20261110-94dd95"},{"kind":"note","id":"note-20261110-4db192"}],"title":"fraise-web views migrated: tokenization workbench built; destinations of the other views","created_at":"2026-11-11T04:49:54.250Z","updated_at":"2026-11-11T07:34:56.459Z","created_by":{"kind":"agent","id":"1fe509bf62d28f74"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":3,"summary":"Follow-up to onboarding, which had wrongly left the repository's own web app out: each of its views now has a destination in AblationKit. The tokenization view was rebuilt as a project panel that runs fraise's own tokenizer and hierarchy code in the project's environment, and its output matched fraise's direct output exactly on the texts checked; training and run inspection map to AblationKit jobs and run pages. The two generation views were not built, because generating 48 tokens from each of the three pilot checkpoints took about a minute on CPU, and giving extensions a GPU is the researcher's decision."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# fraise-web views migrated: tokenization workbench built; destinations of the other views

_ID `note-20261111-544ed3` · tags onboarding, extension, visualisation · agent, 2026-11-11T04:49:54.250Z_

Links:

- report `rep-20261110-94dd95`
- note `note-20261110-4db192`

Follow-up to onboarding, which had wrongly left the repository's own web app out: each of its views now has a destination in AblationKit. The tokenization view was rebuilt as a project panel that runs fraise's own tokenizer and hierarchy code in the project's environment, and its output matched fraise's direct output exactly on the texts checked; training and run inspection map to AblationKit jobs and run pages. The two generation views were not built, because generating 48 tokens from each of the three pilot checkpoints took about a minute on CPU, and giving extensions a GPU is the researcher's decision.

<!-- ablationkit:body -->
Follow-up to the onboarding report `rep-20261110-94dd95` and its note `note-20261110-4db192`. The onboarding left `src/fraise/web` out as "not run outputs", which was wrong: each of its views needed a destination. The onboarding task stays closed.

## Built: `tokenization-workbench`
Build `eb-8834fcd92e3bb162`, active (enabled as built). It is a `project.panel` (Project → Overview), plus two `project.analysis` Python contributions (`catalog`, `tokenize`).
- **How it runs:** in fraise's own environment (`python.environment: "project"`), built from commit 72b1efb; Python 3.14.8, torch 2.14.0, tokenizers 0.23.2.
- **What it calls:** `fraise.web.catalog.Catalog` and `fraise.web.tokenization.TokenizationWorkbench.analyze` (thus `fraise.preview.hierarchy.analyze_text`, `fraise.hierarchy.runs.run_hierarchy`, `two_level_hierarchy`, `hierarchy_from_macro_tokenizer`), over `$ABLATIONKIT_CHECKOUT/data`. It reimplements none of it.
- **Validation:** requests are checked against the bounds of `fraise.web.schemas.TokenizeRequest`. That module itself is not importable, because pydantic belongs to fraise's optional `web` extra.
- **Defaults:** the three pilot runs, each with the hierarchy its own run was trained with.
  - `pilot-base`: plain 8k BPE.
  - `pilot-symmetric`: 64k macro, `independent` embeddings, so no hierarchy.
  - `pilot-fraise`: two-level 8k → 64k with 32-d residuals.
  - All other configuration kinds of the web app are available: tokenizer default, merge-prefix levels, hierarchy file, saved run, segmentation only.
- **The view** ports `static/tokens.js`: segmentation strips, composition icicle, levels/residual table, composition trees and node inspector, with the cards side by side.
- **Capability:** `project.analysis.execute` only (enforced; 120 s, 2048 MiB, 1 CPU, no GPU, network none, all enforced). Filesystem reads are advisory, because the Python is trusted code under the researcher's user.
- **Size limit:** the bridge carries at most 256 KiB of result data, so the analysis travels packed (columnar) and gzipped, losslessly. Measured with three configurations: about 30 bytes per character for a prose README excerpt (~8,300 characters fit) and 47 bytes per character for the code-heavy first 15,000 characters of the README (~5,300 fit). The web app allows 20,000; longer text is refused with a clear message.

**Checks, on real text:**
- `xe-8d76db892edc0cc0` (catalog): 27 tokenizers, 23 runs, 0 hierarchy files; 18.3 s.
- `xe-cb4ea3ab5c7adeb4` (report §3.2.2 sentence, with BOS/EOS): 28 / 16 / 16 positions (26 / 14 / 14 plus the two specials), the report's macro segmentation (`' strawberry'`, `' in the United States'`, `' 1,'`, `'250'`); 14.9 s.
- `xe-185645088c065d0d` (708 characters of README, four configurations including merges 8k → 16k → 64k): 281 / 173 / 173 / 173 positions over 281 base tokens. Fraise has 2,097,152 residual parameters (65,536 × 32), and the three-level hierarchy shows carried nodes. 12.6 s, 41,956-byte payload.
- The view's decoder (`ui/src/codec.ts`) was run on that real payload and on three local texts (prose, README, Unicode/emoji/code). The decoded result equals fraise's direct output exactly.
- Too-long text (15,000 characters: 705,508 bytes compressed), empty text and an unknown kind are all refused with clear errors.
- Not checked: the panel's rendering in a browser. Types pass a strict `tsc` against the SDK.

## The other views of `src/fraise/web`
| fraise-web view | What it does | AblationKit destination | Built |
| --- | --- | --- | --- |
| Tokenization (`#tokens`) | segmentation + hierarchy composition | `tokenization-workbench` panel | yes |
| Compare (`#compare`) | one prompt, sampling settings and seed to several checkpoints; speed, load time, validation | the same pattern: a project-environment extension driving `fraise.web.generation.GenerationStream` / `InferenceSession` | no (below) |
| Autocomplete (`#autocomplete`) | generate from the text before the cursor; accept or discard into a document | generation as for Compare (one checkpoint); the editor's insert/accept UI would be extra | no (below) |
| Run inspector (inside Autocomplete/Compare) | model dimensions, vocabulary, context, step, validation | the AblationKit run page (params, latest metrics, `validation/bits_per_byte` history) for the 12 imported runs and tracked jobs | not needed |
| Train (`#train`) + training monitor | forms/JSON settings, launch or resume, live loss/logs, Stop | AblationKit jobs (`train … --events-file`, as in the onboarding note) and the run page (live metrics, console) | replaced |
| Saved inputs, history sidebar | browser-local convenience | none needed | — |

- **Why generation was not built:** a local test of the web app's own `GenerationStream` on CPU (the three pilot checkpoints, 48 tokens each, greedy) worked, but took 60.6 s at 2.2–3.9 tokens/s and 2.5 GB using all cores. Extension Python gets one core and no GPU unless the manifest asks for `python.gpu: true` (advisory), which is the researcher's decision. Results also return only at the end; there is no token streaming. Not cheap, so it is left for the researcher to decide.
- **Train gap:** cancelling a job stops it; it does not save a resumable checkpoint the way the web app's Stop (stop file) does. Progress since the last checkpoint is lost.

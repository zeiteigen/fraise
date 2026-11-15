<!-- ablationkit:record {"schema_version":1,"record_id":"note-20261110-e157db","project_id":"fraise","kind":"note","tags":["untracked-result","investigation"],"links":[{"kind":"task","id":"task-20261110-apoxgb"}],"title":"Investigation of the overfitthousand checkpoint (2026-09-29), untracked","created_at":"2026-11-10T20:52:45.509Z","updated_at":"2026-11-11T07:34:56.452Z","created_by":{"kind":"agent","id":"6830bb2a421e119b"},"updated_by":{"kind":"agent","id":"abe7acfc81b4142d"},"revision":2,"summary":"A pointer to an investigation made before AblationKit (2026-09-29, kept in the repository, not re-run) into why a small scratch model that had memorised its training text still produced a wrong continuation for a memorised prompt. Its finding: the model errs when given short context and generation feeds the error back, whereas training always supplies the correct context; the beginning-of-sequence token the generation API prepends also changes the continuation, but removing it alone does not fix the example. The scratch run itself was not imported."} -->
<!-- Exported by AblationKit from the coordinator's research records. Edits here are not read back: use the UI or `ablationkit research`. -->

# Investigation of the overfitthousand checkpoint (2026-09-29), untracked

_ID `note-20261110-e157db` · tags untracked-result, investigation · agent, 2026-11-10T20:52:45.509Z_

Links:

- task `task-20261110-apoxgb`

A pointer to an investigation made before AblationKit (2026-09-29, kept in the repository, not re-run) into why a small scratch model that had memorised its training text still produced a wrong continuation for a memorised prompt. Its finding: the model errs when given short context and generation feeds the error back, whereas training always supplies the correct context; the beginning-of-sequence token the generation API prepends also changes the continuation, but removing it alone does not fix the example. The scratch run itself was not imported.

<!-- ablationkit:body -->
Pointer only. `data/investigation/report.md` (with `audit.py`, `prompt_audit.py`, `prefix_audit.py` and their `*.log` / `*_results.json`) analyses why the scratch run `data/models/overfitthousand` produced a wrong continuation for a memorised prompt. Its summary: the checkpoint memorised much of its corpus but still errs with short context, and generation feeds the error back, while training always supplies the correct context. The generation API also prepends BOS, which changes the continuation, but removing BOS alone does not fix the example.

The run itself was not imported (the researcher's choice), and the investigation did not run under AblationKit. Related implementation report: `data/diagnostics/document_training/implementation_report.md` (document-aware training).

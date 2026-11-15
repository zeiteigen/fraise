/**
 * fraise-web's Tokenization view as a project panel. The view only collects input and draws;
 * every analysis runs fraise's own code (python/workbench.py → fraise.workbench.tokenization) in the
 * project's environment.
 */
import { Button, mountReact, Notice, useHost, useHostContext } from "@ablationkit/extension-sdk/react";
import { useEffect, useState } from "react";
import { type Analysis, Card } from "./Card.tsx";
import { decode } from "./codec.ts";
import { CSS } from "./styles.ts";

interface CatalogData {
  data_dir: string;
  tokenizers: { id: string; label: string; vocab_size: number; macro: boolean; base_vocab_size: number }[];
  runs: { id: string; name: string; architecture?: string; checkpoints: string[]; default_checkpoint?: string; error?: string }[];
  hierarchies: { id: string; label: string }[];
  defaults: { label: string; hierarchy: { kind: string; run?: { run_id: string } } }[];
}

type Kind = "auto" | "merges" | "file" | "run" | "none";
const KINDS: [Kind, string][] = [
  ["run", "Saved run (trained hierarchy)"],
  ["auto", "Tokenizer default (two-level)"],
  ["merges", "Merge-prefix levels"],
  ["file", "Hierarchy file"],
  ["none", "Segmentation only"],
];

interface Row {
  key: number;
  label: string;
  kind: Kind;
  tokenizer: string;
  levels: string;
  residuals: string;
  file: string;
  run: string;
  checkpoint: string;
}

const numbers = (value: string): number[] | null => {
  const parts = value.split(/[\s,→>-]+/).filter(Boolean).map(Number);
  return parts.length && parts.every(Number.isInteger) ? parts : null;
};

let nextKey = 1;
const emptyRow = (over: Partial<Row> = {}): Row => ({
  key: nextKey++,
  label: "",
  kind: "run",
  tokenizer: "",
  levels: "",
  residuals: "32",
  file: "",
  run: "",
  checkpoint: "",
  ...over,
});

function request(rows: Row[]) {
  return rows.map((row) => {
    const hierarchy: Record<string, unknown> = { kind: row.kind };
    if (row.kind === "merges") hierarchy["level_sizes"] = numbers(row.levels);
    if (row.kind === "merges" || row.kind === "auto") hierarchy["residual_dims"] = numbers(row.residuals);
    if (row.kind === "file") hierarchy["file_id"] = row.file || null;
    if (row.kind === "run") hierarchy["run"] = row.run ? { run_id: row.run, checkpoint_id: row.checkpoint || null } : null;
    return {
      label: row.label || null,
      tokenizer_id: row.kind === "run" ? null : row.tokenizer || null,
      hierarchy,
    };
  });
}

function Workbench() {
  const host = useHost();
  const ctx = useHostContext();
  const [catalog, setCatalog] = useState<CatalogData>();
  const [rows, setRows] = useState<Row[]>([]);
  const [text, setText] = useState("The strawberry fields were unbelievably photosynthesizing.");
  const [special, setSpecial] = useState(false);
  const [results, setResults] = useState<Analysis[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (ctx.preview) return;
    host.runPython("catalog").then(
      (view) => {
        if (view.result?.status !== "ok") throw new Error(view.result?.errors.join("; ") || view.error || "catalog failed");
        const data = view.result.data as unknown as CatalogData;
        setCatalog(data);
        const macro = [...data.tokenizers].sort((a, b) => Number(b.macro) - Number(a.macro) || b.vocab_size - a.vocab_size)[0];
        setRows(
          data.defaults.map((d) =>
            emptyRow({ label: d.label, kind: d.hierarchy.kind as Kind, run: d.hierarchy.run?.run_id ?? "", tokenizer: macro?.id ?? "" }),
          ),
        );
      },
      (e: Error) => setError(e.message),
    );
  }, [host, ctx.preview]);

  const analyze = async () => {
    if (!text || !rows.length) return;
    setBusy(true);
    setError(undefined);
    try {
      const view = await host.runPython("tokenize", { text, add_special_tokens: special, configurations: request(rows) });
      if (view.result?.status !== "ok")
        throw new Error(view.result?.errors.join("; ") || view.error || `analysis ${view.status}`);
      const decoded = (await decode(view.result.data as { encoding?: string; payload?: string })) as { results: Analysis[] };
      setResults(decoded.results);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const update = (key: number, change: Partial<Row>) => setRows((rs) => rs.map((r) => (r.key === key ? { ...r, ...change } : r)));

  if (ctx.preview) return <Notice>Preview: the workbench runs fraise's Python once the build is enabled.</Notice>;
  if (!catalog)
    return (
      <div className="tw-root">
        <style>{CSS}</style>
        {error ? <Notice kind="error">Could not read fraise's catalog: {error}</Notice> : <Notice>Loading fraise's tokenizers and runs…</Notice>}
      </div>
    );

  return (
    <div className="tw-root" data-testid="tokenization-workbench">
      <style>{CSS}</style>
      <p className="tw-help">
        Segmentation decides transformer positions; composition only builds each input token's embedding from smaller
        tokens. Computed by fraise's own <code>fraise.workbench.tokenization</code> over <code>{catalog.data_dir}</code>.
      </p>
      <div className="tw-panel">
        <div className="tw-heading">
          <h2>Text</h2>
          <label className="tw-check">
            <input type="checkbox" checked={special} onChange={(e) => setSpecial(e.target.checked)} /> Include BOS/EOS
          </label>
        </div>
        <textarea
          className="tw-text"
          aria-label="Text to tokenize"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void analyze();
          }}
        />
        <div className="tw-heading">
          <h2>Configurations</h2>
          <button
            type="button"
            className="tw-link"
            disabled={rows.length >= 6}
            onClick={() => setRows((rs) => [...rs, emptyRow({ kind: "auto", tokenizer: rs[0]?.tokenizer ?? catalog.tokenizers[0]?.id ?? "" })])}
          >
            Add configuration to compare
          </button>
        </div>
        <div className="tw-configs">
          {rows.map((row) => {
            const run = catalog.runs.find((r) => r.id === row.run);
            return (
              <div key={row.key} className="tw-config">
                <label>
                  Label
                  <input value={row.label} placeholder="optional" onChange={(e) => update(row.key, { label: e.target.value })} />
                </label>
                <label>
                  Hierarchy
                  <select value={row.kind} onChange={(e) => update(row.key, { kind: e.target.value as Kind })}>
                    {KINDS.map(([v, l]) => (
                      <option key={v} value={v}>
                        {l}
                      </option>
                    ))}
                  </select>
                </label>
                {row.kind === "run" ? (
                  <>
                    <label>
                      Run
                      <select value={row.run} onChange={(e) => update(row.key, { run: e.target.value, checkpoint: "" })}>
                        <option value="">Select a run</option>
                        {catalog.runs
                          .filter((r) => !r.error)
                          .map((r) => (
                            <option key={r.id} value={r.id}>
                              {r.name} · {r.architecture}
                            </option>
                          ))}
                      </select>
                    </label>
                    <label>
                      Checkpoint
                      <select value={row.checkpoint} onChange={(e) => update(row.key, { checkpoint: e.target.value })}>
                        <option value="">Default ({run?.default_checkpoint ?? "—"})</option>
                        {(run?.checkpoints ?? []).map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </label>
                  </>
                ) : (
                  <label>
                    Tokenizer
                    <select value={row.tokenizer} onChange={(e) => update(row.key, { tokenizer: e.target.value })}>
                      {catalog.tokenizers.map((t) => (
                        <option key={t.id} value={t.id}>
                          {t.label} ·{" "}
                          {t.macro
                            ? `macro ${t.vocab_size.toLocaleString()} over base ${t.base_vocab_size.toLocaleString()}`
                            : `plain BPE ${t.vocab_size.toLocaleString()}`}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {row.kind === "merges" ? (
                  <label>
                    Level sizes
                    <input value={row.levels} placeholder="8192 16384 65536" onChange={(e) => update(row.key, { levels: e.target.value })} />
                  </label>
                ) : null}
                {row.kind === "merges" || row.kind === "auto" ? (
                  <label>
                    Residual widths
                    <input value={row.residuals} placeholder="32" onChange={(e) => update(row.key, { residuals: e.target.value })} />
                  </label>
                ) : null}
                {row.kind === "file" ? (
                  <label>
                    Hierarchy file
                    <select value={row.file} onChange={(e) => update(row.key, { file: e.target.value })}>
                      <option value="">{catalog.hierarchies.length ? "Select a file" : "No files in DATA_DIR/hierarchies"}</option>
                      {catalog.hierarchies.map((h) => (
                        <option key={h.id} value={h.id}>
                          {h.label}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <button
                  type="button"
                  className="tw-link"
                  disabled={rows.length <= 1}
                  onClick={() => setRows((rs) => rs.filter((r) => r.key !== row.key))}
                >
                  Remove
                </button>
              </div>
            );
          })}
        </div>
        <div className="tw-footer">
          <p className="tw-help">
            Level sizes start at the base vocabulary, e.g. <code>8192 16384 65536</code>; residual widths are one per
            composed level. Ctrl/Cmd+Enter analyzes. fraise's Python stays loaded between analyses; the first one after a while takes a few seconds longer.
          </p>
          <Button onClick={() => void analyze()} disabled={busy || !text}>
            {busy ? "Analyzing…" : "Analyze"}
          </Button>
        </div>
        {error ? <Notice kind="error">{error}</Notice> : null}
      </div>
      <div className="tw-results">
        {results.map((r, i) => (
          <Card key={`${i}-${r.name}`} result={r} index={i} />
        ))}
      </div>
    </div>
  );
}

void mountReact(Workbench);

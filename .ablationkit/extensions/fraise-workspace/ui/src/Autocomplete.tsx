/**
 * fraise-web's Autocomplete view: edit a document, continue the text before the cursor with a
 * saved checkpoint, then accept the continuation at that cursor or discard it. The model gets the
 * raw text (no roles or chat templates). Generation runs python/workspace.py:generate.
 */
import { mountReact, Notice, useHost } from "@ablationkit/extension-sdk/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  CheckpointOptions,
  copyText,
  DEFAULT_SETTINGS,
  formatNumber,
  type GenerationResult,
  generate,
  type Json,
  metricsText,
  parseRef,
  partialText,
  type Ref,
  RunInformation,
  refKey,
  run,
  runDecoded,
  type Settings,
  SettingsEditor,
  type Streaming,
  useRuns,
} from "./common.tsx";
import { CSS } from "./styles.ts";

interface SavedSummary {
  id: string;
  title: string;
  run_id: string;
  checkpoint_id: string | null;
  updated: number;
}

interface SavedItem {
  id: string;
  ref: Ref;
  text: string;
  start: number;
  end: number;
  suggestion: string;
  metrics: GenerationResult | null;
  settings: Settings;
}

function Autocomplete() {
  const host = useHost();
  const { data, error: loadError, preview, reload } = useRuns();
  const editor = useRef<HTMLTextAreaElement>(null);
  const [selected, setSelected] = useState("");
  const [text, setText] = useState("");
  const [cursor, setCursor] = useState({ start: 0, end: 0 });
  const [pendingCursor, setPendingCursor] = useState<number | { start: number; end: number }>();
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [suggestion, setSuggestion] = useState("");
  const [streaming, setStreaming] = useState<Streaming>();
  const [metrics, setMetrics] = useState<GenerationResult | null>(null);
  const [replacement, setReplacement] = useState<{ text: string; start: number; end: number } | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [saved, setSaved] = useState<SavedSummary[]>([]);
  const [savedId, setSavedId] = useState<string | null>(null);

  const runs = data?.runs ?? [];
  const ref = parseRef(selected);
  const current = runs.find((r) => r.id === ref?.run_id);
  const checkpoint = current?.checkpoints.find((c) => c.id === ref?.checkpoint_id);

  useEffect(() => {
    if (!data || selected) return;
    const first = data.runs.find((r) => r.checkpoints.length);
    if (first)
      setSelected(refKey({ run_id: first.id, checkpoint_id: first.default_checkpoint ?? first.checkpoints[0]?.id ?? "" }));
  }, [data, selected]);

  useEffect(() => {
    if (preview) return;
    run(host, "inputs", { action: "list" }).then(
      (r) => setSaved((r.data as { items: SavedSummary[] }).items),
      () => undefined,
    );
  }, [host, preview]);

  // Put the cursor where an accept or restore left it, once the new text is rendered.
  useLayoutEffect(() => {
    if (pendingCursor === undefined || !editor.current) return;
    const { start, end } = typeof pendingCursor === "number" ? { start: pendingCursor, end: pendingCursor } : pendingCursor;
    editor.current.focus();
    editor.current.setSelectionRange(start, end);
    setCursor({ start, end });
    setPendingCursor(undefined);
  }, [pendingCursor]);

  const say = (message: string) => {
    setNotice(message);
    setTimeout(() => setNotice((n) => (n === message ? undefined : n)), 4000);
  };

  const readCursor = () => {
    const el = editor.current;
    if (el) setCursor({ start: el.selectionStart, end: el.selectionEnd });
  };

  const discard = () => {
    if (running) return;
    setSuggestion("");
    setStreaming(undefined);
    setMetrics(null);
    setReplacement(null);
    setError(undefined);
  };

  const save = async (item: Omit<SavedItem, "id">) => {
    try {
      const r = await run(host, "inputs", { action: "save", item: { ...item, id: savedId } as unknown as Json });
      const d = r.data as { items: SavedSummary[]; saved: string };
      setSaved(d.items);
      setSavedId(d.saved);
    } catch (e) {
      say(`Could not save this input: ${(e as Error).message}`);
    }
  };

  const complete = async () => {
    const start = editor.current?.selectionStart ?? cursor.start;
    const end = editor.current?.selectionEnd ?? cursor.end;
    if (!ref || !start || running) return;
    discard();
    const replace = { text, start, end };
    setReplacement(replace);
    setRunning(true);
    let final = "";
    let result: GenerationResult | null = null;
    try {
      const outcome = await generate(host, { models: [ref], prompt: text.slice(0, start), ...settings }, (p) => {
        setStreaming(p);
        setSuggestion(partialText(p));
      });
      if (outcome.error) throw new Error(outcome.error.message || "Generation failed.");
      result = outcome.results[0] ?? null;
      final = result?.text ?? "";
      setSuggestion(final);
      setMetrics(result);
      setStreaming(undefined);
    } catch (e) {
      setError((e as Error).message);
      setStreaming(undefined);
      setSuggestion("");
    } finally {
      setRunning(false);
    }
    await save({ ref, text, start: replace.start, end: replace.end, suggestion: final, metrics: result, settings });
    editor.current?.focus();
  };

  const accept = () => {
    if (running || !suggestion || !replacement || streaming) return;
    const { text: base, start, end } = replacement;
    if (text !== base) return discard();
    const next = base.slice(0, start) + suggestion + base.slice(end);
    const position = start + suggestion.length;
    setText(next);
    discard();
    setPendingCursor(position);
    if (ref) void save({ ref, text: next, start: position, end: position, suggestion: "", metrics: null, settings });
  };

  const restore = async (id: string) => {
    if (running) return say("Wait for generation to finish before switching inputs.");
    try {
      const item = await runDecoded<SavedItem>(host, "inputs", { action: "get", id });
      discard();
      setSavedId(item.id);
      setText(item.text);
      const key = refKey({ run_id: item.ref.run_id, checkpoint_id: item.ref.checkpoint_id });
      const available = runs.some((r) => r.id === item.ref.run_id && r.checkpoints.some((c) => c.id === item.ref.checkpoint_id));
      setSelected(available ? key : "");
      setSettings({ ...DEFAULT_SETTINGS, ...item.settings });
      setSuggestion(item.suggestion || "");
      setMetrics(item.metrics || null);
      setReplacement(item.suggestion ? { text: item.text, start: item.start, end: item.end } : null);
      setPendingCursor({ start: item.start ?? item.text.length, end: item.end ?? item.text.length });
      if (!available) setError("The saved checkpoint is no longer available. Select another checkpoint to generate.");
    } catch (e) {
      say((e as Error).message);
    }
  };

  const newInput = () => {
    if (running) return say("Wait for generation to finish before starting a new input.");
    discard();
    setSavedId(null);
    setText("");
    setPendingCursor(0);
  };

  const clearSaved = async () => {
    if (running) return say("Wait for generation to finish before clearing saved inputs.");
    await run(host, "inputs", { action: "clear" });
    setSaved([]);
    setSavedId(null);
    say("Saved inputs cleared");
  };

  const release = async () => {
    try {
      const r = await run(host, "release");
      say(r.summary);
    } catch (e) {
      say((e as Error).message);
    }
  };

  if (preview) return <Notice>Preview: Autocomplete runs fraise's Python once the build is enabled.</Notice>;
  if (!data)
    return (
      <div className="fw-root">
        <style>{CSS}</style>
        {loadError ? <Notice kind="error">Could not read fraise's runs: {loadError}</Notice> : <Notice>Loading saved checkpoints…</Notice>}
      </div>
    );

  const canGenerate = !running && Boolean(ref) && cursor.start > 0;
  const status = running
    ? streaming?.phase === "loading"
      ? "Loading checkpoint…"
      : "Generating…"
    : !selected
      ? `No checkpoint available in ${data.models_dir}. Train a model first.`
      : cursor.start === 0
        ? "Enter text and place the cursor after the context."
        : "Ready";

  return (
    <div className="fw-root" data-testid="fraise-autocomplete">
      <style>{CSS}</style>
      <p className="fw-help">
        Generate a continuation from the text before the cursor, then accept it to insert it at that cursor. Computed by
        fraise's own <code>fraise.workbench</code> on {data.device.resolved.toUpperCase()} from runs in{" "}
        <code>{data.models_dir}</code>.
      </p>
      <div className="fw-layout wide-side">
        <aside className="fw-panel fw-side" aria-label="Saved inputs">
          <div className="fw-heading">
            <h2>Saved inputs</h2>
            <button type="button" className="fw-link" onClick={() => void clearSaved()} disabled={!saved.length}>
              Clear
            </button>
          </div>
          <button type="button" className="fw-button" onClick={newInput} style={{ width: "100%", marginBottom: 8 }}>
            New input
          </button>
          <div className="fw-saved">
            {saved.length ? null : <p className="fw-help">No saved inputs.</p>}
            {saved.map((item) => (
              <button
                type="button"
                key={item.id}
                className={item.id === savedId ? "active" : ""}
                title={item.checkpoint_id ?? item.run_id}
                onClick={() => void restore(item.id)}
              >
                <span>{item.title || "Untitled input"}</span>
                <small>{runs.find((r) => r.id === item.run_id)?.name ?? item.run_id}</small>
              </button>
            ))}
          </div>
        </aside>
        <div>
          <form
            className="fw-panel"
            onSubmit={(e) => {
              e.preventDefault();
              void complete();
            }}
          >
            <div className="fw-heading">
              <h2>Input text</h2>
              <label className="fw-field">
                Checkpoint
                <select
                  value={selected}
                  disabled={running}
                  onChange={(e) => {
                    setSelected(e.target.value);
                    setSavedId(null);
                    discard();
                  }}
                >
                  <option value="">Select a checkpoint</option>
                  <CheckpointOptions runs={runs} />
                </select>
              </label>
            </div>
            <textarea
              ref={editor}
              className="fw-document"
              aria-label="Text to complete"
              maxLength={100000}
              spellCheck={false}
              placeholder="Enter text, place the cursor, then generate a continuation."
              value={text}
              disabled={running}
              onChange={(e) => {
                setText(e.target.value);
                discard();
                readCursor();
              }}
              onSelect={readCursor}
              onClick={readCursor}
              onKeyUp={readCursor}
              onKeyDown={(e) => {
                if (e.nativeEvent.isComposing) return;
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  if (canGenerate) void complete();
                } else if (e.key === "Tab" && !e.shiftKey && suggestion && !running) {
                  e.preventDefault();
                  accept();
                } else if (e.key === "Escape" && suggestion && !running) {
                  e.preventDefault();
                  discard();
                }
              }}
            />
            <div className="fw-status">
              <span>
                {formatNumber(text.length, 0)} characters · cursor {formatNumber(cursor.start, 0)}
                {cursor.end > cursor.start ? ` · ${cursor.end - cursor.start} selected` : ""}
              </span>
              <span>Ctrl/Cmd+Enter: generate · Tab: accept · Esc: discard</span>
            </div>
            <SettingsEditor value={settings} onChange={setSettings} disabled={running} open />
            <div className="fw-toolbar">
              <span role="status" className="fw-help">
                {status}
              </span>
              <div className="fw-buttons">
                <button type="button" className="fw-link" onClick={() => void release()} disabled={running} title="Unload the checkpoint and free its device memory, e.g. before training">
                  Release device memory
                </button>
                <button type="submit" className="fw-button primary" disabled={!canGenerate}>
                  {running ? "Generating…" : "Generate continuation"}
                </button>
              </div>
            </div>
            {running && streaming ? (
              <div className="fw-progress" aria-hidden="true">
                <div style={{ width: `${streaming.percent}%` }} />
              </div>
            ) : null}
            {error ? <p className="fw-error" role="alert">{error}</p> : null}
          </form>
          <section className="fw-panel" aria-label="Suggested continuation">
            <div className="fw-heading">
              <h2>Continuation</h2>
              <span className="fw-metrics">
                {streaming ? `${streaming.tokens} generated tokens` : metrics ? `${metrics.generated_tokens} generated tokens` : ""}
              </span>
            </div>
            <pre className="fw-output" aria-label="Generated continuation">
              {suggestion}
            </pre>
            <p className="fw-metrics">{streaming ? metricsText(streaming) : metrics ? metricsText(metrics) : "No continuation generated."}</p>
            <div className="fw-toolbar">
              <span className="fw-help">Accept inserts at the cursor position used for generation, replacing any selection.</span>
              <div className="fw-buttons">
                <button type="button" className="fw-button" disabled={running || !suggestion} onClick={() => void copyText(suggestion, say)}>
                  Copy continuation
                </button>
                <button type="button" className="fw-button" disabled={running || !suggestion} onClick={discard}>
                  Discard
                </button>
                <button type="button" className="fw-button primary" disabled={running || !suggestion || Boolean(streaming)} onClick={accept}>
                  Accept continuation
                </button>
              </div>
            </div>
            {notice ? <p className="fw-ok" role="status">{notice}</p> : null}
          </section>
        </div>
        <div>
          <RunInformation run={current} checkpoint={checkpoint} device={data.device} />
          <button type="button" className="fw-link" onClick={reload} disabled={running}>
            Refresh runs
          </button>
        </div>
      </div>
    </div>
  );
}

void mountReact(Autocomplete);

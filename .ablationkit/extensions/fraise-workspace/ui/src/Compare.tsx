/**
 * fraise-web's Compare view: the same prompt, sampling settings and seed sent to up to eight
 * checkpoints, which run one after another (python/workspace.py:generate).
 */
import { mountReact, Notice, useHost } from "@ablationkit/extension-sdk/react";
import { useState } from "react";
import {
  copyText,
  DEFAULT_SETTINGS,
  formatNumber,
  type GenerationResult,
  generate,
  metricsText,
  parseRef,
  partialText,
  type Ref,
  type Run,
  refKey,
  type Settings,
  SettingsEditor,
  type Streaming,
  useRuns,
} from "./common.tsx";
import { CSS } from "./styles.ts";

interface CardState {
  ref: Ref;
  status: "Queued" | "Loading" | "Generating" | "Finished" | "Complete" | "Failed" | "Interrupted";
  text: string;
  metrics: string;
}

const BADGE: Record<CardState["status"], string> = {
  Queued: "",
  Loading: "running",
  Generating: "running",
  Finished: "success",
  Complete: "success",
  Failed: "failed",
  Interrupted: "",
};

function RunDetails(props: { run: Run }) {
  const { run } = props;
  const data = {
    model_config: run.config,
    training_config: run.training,
    validation: run.validation,
    validation_results: run.validation_results,
  };
  return (
    <details className="fw-details">
      <summary>Saved model, tokenizer &amp; validation</summary>
      <dl className="fw-info">
        <div>
          <dt>Tokenizer</dt>
          <dd>{run.tokenizer?.id ?? "Archived tokenizer"}</dd>
        </div>
        <div>
          <dt>Vocabulary</dt>
          <dd>{run.tokenizer?.vocab_size ? `${formatNumber(run.tokenizer.vocab_size, 0)} tokens` : "Unavailable"}</dd>
        </div>
        <div>
          <dt>Fingerprint</dt>
          <dd>{run.tokenizer?.fingerprint ?? "Unavailable"}</dd>
        </div>
      </dl>
      <p className="fw-help">Validation loss and token counts depend on the saved tokenizer.</p>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

function Compare() {
  const host = useHost();
  const { data, error: loadError, preview, reload } = useRuns();
  const [chosen, setChosen] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("");
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [cards, setCards] = useState<CardState[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();

  const say = (message: string) => {
    setNotice(message);
    setTimeout(() => setNotice((n) => (n === message ? undefined : n)), 4000);
  };

  if (preview) return <Notice>Preview: Compare runs fraise's Python once the build is enabled.</Notice>;
  if (!data)
    return (
      <div className="fw-root">
        <style>{CSS}</style>
        {loadError ? <Notice kind="error">Could not read fraise's runs: {loadError}</Notice> : <Notice>Loading saved checkpoints…</Notice>}
      </div>
    );

  const runs = data.runs.filter((r) => r.checkpoints.length);
  const toggle = (key: string, on: boolean) => {
    if (on && chosen.length >= 8) return say("Choose up to eight checkpoints per comparison.");
    setChosen((c) => (on ? [...c, key] : c.filter((k) => k !== key)));
  };

  const update = (index: number, change: Partial<CardState>) =>
    setCards((cs) => cs.map((c, i) => (i === index ? { ...c, ...change } : c)));

  const send = async () => {
    const models = chosen.map(parseRef).filter((r): r is Ref => r !== null);
    if (!models.length || !prompt.trim() || running) return;
    setError(undefined);
    setCards(models.map((ref) => ({ ref, status: "Queued", text: "", metrics: "Waiting for earlier checkpoints" })));
    setRunning(true);
    let seen = -1;
    try {
      const outcome = await generate(host, { models, prompt, ...settings }, (p: Streaming) => {
        // A checkpoint has finished once the next one reports; its full output and timings
        // arrive with the comparison's result.
        if (p.index > seen) {
          for (let i = Math.max(seen, 0); i < p.index; i++)
            update(i, { status: "Finished", metrics: "Finished · full output and timings when the comparison ends" });
          seen = p.index;
        }
        update(p.index, {
          status: p.phase === "loading" ? "Loading" : "Generating",
          text: partialText(p),
          metrics: p.phase === "loading" ? "Loading checkpoint…" : metricsText(p),
        });
      });
      setCards((cs) =>
        cs.map((card, index) => {
          const result = outcome.results.find((r: GenerationResult) => r.index === index);
          if (result) return { ...card, status: "Complete", text: result.text, metrics: metricsText(result) };
          if (outcome.error && outcome.error.index === index)
            return { ...card, status: "Failed", metrics: outcome.error.message || "Generation failed" };
          return { ...card, status: "Interrupted", metrics: card.text ? "Partial output kept" : "Could not finish generation" };
        }),
      );
      if (outcome.error && outcome.error.index == null) setError(outcome.error.message);
    } catch (e) {
      setError((e as Error).message);
      setCards((cs) =>
        cs.map((card) =>
          card.status === "Complete"
            ? card
            : { ...card, status: "Interrupted", metrics: card.text ? "Partial output kept" : "Could not finish generation" },
        ),
      );
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="fw-root" data-testid="fraise-compare">
      <style>{CSS}</style>
      <p className="fw-help">
        Run the same text and sampling settings across selected checkpoints, one after another with the same seed.
        Computed by fraise's own <code>fraise.workbench</code> on {data.device.resolved.toUpperCase()} from runs in{" "}
        <code>{data.models_dir}</code>.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <div className="fw-panel">
          <div className="fw-heading">
            <h2>Checkpoints</h2>
            <span className="fw-buttons">
              <button type="button" className="fw-link" onClick={reload} disabled={running}>
                Refresh
              </button>
              <span className="fw-badge">{chosen.length} selected</span>
            </span>
          </div>
          <div className="fw-runs">
            {runs.length ? null : <p className="fw-help">No checkpoints available. Train a model to compare checkpoints.</p>}
            {runs.map((r) => (
              <div key={r.id} className="fw-run">
                <div className="fw-run-heading">
                  <div>
                    <strong>{r.name || r.id}</strong>
                    <small>
                      {r.architecture ?? "Unknown architecture"} · {formatNumber(r.parameter_count, 0)} parameters · context{" "}
                      {formatNumber(r.config?.max_seq_len, 0)} · vocab {formatNumber(r.tokenizer?.vocab_size, 0)} · saved step{" "}
                      {formatNumber(r.global_step ?? 0, 0)}
                    </small>
                  </div>
                  <span className="fw-badge">{r.finished ? "Finished" : "Saved run"}</span>
                </div>
                <div className="fw-chips">
                  {r.checkpoints.map((c) => {
                    const key = refKey({ run_id: r.id, checkpoint_id: c.id });
                    return (
                      <label key={key} className="fw-chip">
                        <input
                          type="checkbox"
                          disabled={running}
                          checked={chosen.includes(key)}
                          onChange={(e) => toggle(key, e.target.checked)}
                        />
                        {c.label || c.id}
                      </label>
                    );
                  })}
                </div>
                <RunDetails run={r} />
              </div>
            ))}
          </div>
        </div>
        <div className="fw-panel">
          <div className="fw-heading">
            <h2>Input text</h2>
          </div>
          <textarea
            className="fw-prompt"
            aria-label="Shared input text"
            placeholder="Text to continue…"
            required
            maxLength={100000}
            disabled={running}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.nativeEvent.isComposing) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <SettingsEditor value={settings} onChange={setSettings} disabled={running} />
          <div className="fw-toolbar">
            <p className="fw-help">Checkpoints run sequentially. Ctrl/Cmd+Enter to run; Enter for a newline.</p>
            <button type="submit" className="fw-button primary" disabled={running || !chosen.length || !prompt.trim()}>
              {running ? "Comparing…" : "Compare checkpoints"}
            </button>
          </div>
          {error ? <p className="fw-error" role="alert">{error}</p> : null}
          {notice ? <p className="fw-ok" role="status">{notice}</p> : null}
        </div>
      </form>
      <div className="fw-results" aria-label="Comparison results">
        {cards.map((card, index) => {
          const r = data.runs.find((x) => x.id === card.ref.run_id);
          const c = r?.checkpoints.find((x) => x.id === card.ref.checkpoint_id);
          return (
            <article key={refKey(card.ref)} className={`fw-card${card.status === "Loading" || card.status === "Generating" ? " streaming" : ""}`}>
              <div className="fw-heading">
                <div>
                  <h3>{r?.name ?? card.ref.run_id}</h3>
                  <small>{c?.label ?? card.ref.checkpoint_id}</small>
                </div>
                <span className={`fw-badge ${BADGE[card.status]}`}>{card.status}</span>
              </div>
              <pre className="fw-output">{card.text}</pre>
              <div className="fw-toolbar">
                <span className="fw-metrics">{card.metrics}</span>
                <button
                  type="button"
                  className="fw-link"
                  aria-label={`Copy output from ${r?.name ?? card.ref.run_id}`}
                  disabled={!card.text}
                  onClick={() => void copyText(cards[index]?.text ?? "", say)}
                >
                  Copy
                </button>
              </div>
            </article>
          );
        })}
      </div>
      {cards.length ? (
        <p className="fw-help">
          Each checkpoint uses its saved configuration and tokenizer. Token counts, speed, and validation losses can differ
          with the tokenizer and are not always directly comparable.
        </p>
      ) : null}
    </div>
  );
}

void mountReact(Compare);

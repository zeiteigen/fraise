/**
 * What the Autocomplete, Compare and Train panels share: the runs catalog, checkpoint references,
 * sampling settings, the run information card and the streamed `generate` call. Every result
 * comes from python/workspace.py (fraise.workbench); the views only collect input and draw.
 */
import type { ExecutionView, Host } from "@ablationkit/extension-sdk";
import { useHost, useHostContext } from "@ablationkit/extension-sdk/react";
import { useCallback, useEffect, useState } from "react";
import { decode } from "./codec.ts";

export type Json = Record<string, unknown>;

export interface Checkpoint {
  id: string;
  label: string;
  step: number | null;
  kind: "checkpoint" | "final";
}

export interface Run {
  id: string;
  name: string;
  architecture?: string;
  parameter_count?: number;
  config?: Json & { max_seq_len?: number; architecture?: string };
  training?: Json;
  tokenizer?: { id: string; label: string; vocab_size: number; fingerprint: string };
  checkpoints: Checkpoint[];
  default_checkpoint?: string | null;
  finished?: boolean;
  global_step?: number;
  tokens_seen?: number;
  validation?: [number, number] | { loss?: number } | null;
  validation_results?: { bits_per_byte?: number }[];
  can_resume: boolean;
  error?: string;
}

export interface Device {
  resolved: string;
  cuda_available: boolean;
}

export interface RunsData {
  data_dir: string;
  models_dir: string;
  device: Device;
  runs: Run[];
}

export interface Ref {
  run_id: string;
  checkpoint_id: string;
}

export const refKey = (ref: Ref) => JSON.stringify([ref.run_id, ref.checkpoint_id]);
export function parseRef(value: string): Ref | null {
  try {
    const [run_id, checkpoint_id] = JSON.parse(value) as [string, string];
    return run_id ? { run_id, checkpoint_id } : null;
  } catch {
    return null;
  }
}

export const formatNumber = (value: unknown, digits = 1) =>
  value != null && Number.isFinite(Number(value))
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(Number(value))
    : "—";

/** Run a Python contribution; reject with its error unless it succeeded. */
export async function run(host: Host, target: string, params: Json = {}, onProgress?: (v: ExecutionView) => void) {
  const view = await host.runPython(target, params, onProgress ? { onProgress } : {});
  if (view.result?.status !== "ok")
    throw new Error(view.result?.errors.join("; ") || view.error || `${target} ${view.status}`);
  return view.result;
}

export async function runDecoded<T>(host: Host, target: string, params: Json = {}): Promise<T> {
  const result = await run(host, target, params);
  return (await decode(result.data as { encoding?: string; payload?: string })) as T;
}

/** The saved runs, loaded once per panel and on `reload()`. */
export function useRuns() {
  const host = useHost();
  const ctx = useHostContext();
  const [data, setData] = useState<RunsData>();
  const [error, setError] = useState<string>();
  const reload = useCallback(() => {
    if (ctx.preview) return;
    setError(undefined);
    runDecoded<RunsData>(host, "runs").then(setData, (e: Error) => setError(e.message));
  }, [host, ctx.preview]);
  useEffect(reload, [reload]);
  return { data, error, reload, preview: ctx.preview };
}

export function CheckpointOptions(props: { runs: Run[]; filter?: (c: Checkpoint) => boolean }) {
  return (
    <>
      {props.runs
        .filter((r) => r.checkpoints.some(props.filter ?? (() => true)))
        .map((r) => (
          <optgroup key={r.id} label={r.name || r.id}>
            {r.checkpoints.filter(props.filter ?? (() => true)).map((c) => (
              <option key={c.id} value={refKey({ run_id: r.id, checkpoint_id: c.id })}>
                {c.label || c.id}
              </option>
            ))}
          </optgroup>
        ))}
    </>
  );
}

// ------------------------------------------------------------------------------ sampling settings

export interface Settings {
  max_new_tokens: number | null;
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  seed: number | null;
}

export const DEFAULT_SETTINGS: Settings = { max_new_tokens: 128, temperature: 0, top_p: 0.95, top_k: null, seed: 1337 };

const SETTING_FIELDS: [keyof Settings, string, Json][] = [
  ["max_new_tokens", "Max new tokens", { min: 1, max: 4096, step: 1, required: true }],
  ["temperature", "Temperature", { min: 0, max: 100, step: 0.05, required: true }],
  ["top_p", "Top p", { min: 0.000001, max: 1, step: "any", placeholder: "Optional" }],
  ["top_k", "Top k", { min: 1, step: 1, placeholder: "Optional" }],
  ["seed", "Seed", { min: 0, step: 1, required: true }],
];

export function SettingsEditor(props: { value: Settings; onChange: (s: Settings) => void; disabled?: boolean; open?: boolean }) {
  const v = props.value;
  return (
    <details className="fw-settings" open={props.open}>
      <summary>
        Generation settings{" "}
        <span className="fw-metrics">
          {v.max_new_tokens ?? "—"} tokens · {v.temperature === 0 ? "greedy" : `temp ${v.temperature ?? "—"}`}
        </span>
      </summary>
      <div className="fw-settings-grid">
        {SETTING_FIELDS.map(([name, label, attrs]) => (
          <label key={name} className="fw-field">
            {label}
            <input
              type="number"
              {...attrs}
              disabled={props.disabled}
              value={v[name] ?? ""}
              onChange={(e) => props.onChange({ ...v, [name]: e.target.value === "" ? null : Number(e.target.value) })}
            />
          </label>
        ))}
      </div>
      <p className="fw-help">Temperature 0 uses greedy decoding. Token counts depend on each model’s tokenizer.</p>
    </details>
  );
}

// ------------------------------------------------------------------------------ run information

export function RunInformation(props: { run?: Run; checkpoint?: Checkpoint; device?: Device }) {
  const { run, checkpoint } = props;
  if (!run)
    return (
      <aside className="fw-panel fw-side">
        <h2>Model information</h2>
        <p className="fw-help">Select an available checkpoint.</p>
      </aside>
    );
  const validation = Array.isArray(run.validation) ? run.validation[1] : run.validation?.loss;
  // Bits per byte is comparable across tokenizers; token losses are not.
  const scored = (run.validation_results ?? []).filter((r) => r.bits_per_byte != null);
  const bpb = scored.length ? scored[scored.length - 1]?.bits_per_byte : null;
  const rows: [string, string][] = [
    ["Run", run.name || run.id],
    ["Checkpoint", checkpoint?.label ?? "Unavailable"],
    ["Architecture", run.architecture ?? "—"],
    ["Parameters", formatNumber(run.parameter_count, 0)],
    ["Context limit", `${formatNumber(run.config?.max_seq_len, 0)} tokens`],
    ["Vocabulary", `${formatNumber(run.tokenizer?.vocab_size, 0)} tokens`],
    ["Checkpoint step", formatNumber(checkpoint?.step, 0)],
    ["Run training tokens", formatNumber(run.tokens_seen, 0)],
    ["Latest validation loss", validation == null ? "—" : formatNumber(validation, 4)],
    ["Latest validation bits/byte", bpb == null ? "—" : formatNumber(bpb, 4)],
    ["Device", props.device?.resolved?.toUpperCase() ?? "—"],
  ];
  return (
    <aside className="fw-panel fw-side" aria-label="Selected model information">
      <h2>Model information</h2>
      <dl className="fw-info">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {run.config ? (
        <details className="fw-details">
          <summary>Model configuration</summary>
          <pre>{JSON.stringify(run.config, null, 2)}</pre>
        </details>
      ) : null}
    </aside>
  );
}

// ------------------------------------------------------------------------------ generation

export interface GenerationResult {
  index: number;
  run_id: string;
  checkpoint_id: string;
  text: string;
  generated_tokens: number;
  seconds: number;
  tokens_per_second: number;
  load_seconds: number;
  finish_reason: string | null;
  prompt_tokens: number;
}

export interface GenerationOutcome {
  results: GenerationResult[];
  error: { index: number | null; message: string } | null;
}

/** One model's output while it streams. `offset > 0` means an earlier stretch went by unseen. */
export interface Streaming {
  index: number;
  phase: "loading" | "generating";
  offset: number;
  text: string;
  tokens: number;
  percent: number;
}

export const partialText = (p: Streaming) => (p.offset > 0 ? "…" : "") + p.text;

/**
 * Continue `prompt` with each checkpoint. Progress messages carry the output's length and its end
 * (at most ~480 characters); they are stitched onto what was already seen, and the full outputs
 * arrive with the result.
 */
export async function generate(
  host: Host,
  request: { models: Ref[]; prompt: string } & Settings,
  onPartial: (p: Streaming) => void,
): Promise<GenerationOutcome> {
  let current: Streaming | undefined;
  const result = await run(host, "generate", request as unknown as Json, (view) => {
    const message = view.progress?.message;
    if (!message) return;
    let m: { i: number; phase?: string; n?: number; len?: number; tail?: string };
    try {
      m = JSON.parse(message);
    } catch {
      return;
    }
    const percent = view.progress?.percent ?? 0;
    if (m.phase === "loading") {
      current = { index: m.i, phase: "loading", offset: 0, text: "", tokens: 0, percent };
    } else if (typeof m.tail === "string" && typeof m.len === "number") {
      const prev = current?.index === m.i ? current : { index: m.i, phase: "generating" as const, offset: 0, text: "", tokens: 0, percent };
      const start = m.len - m.tail.length;
      const end = prev.offset + prev.text.length;
      current =
        start >= prev.offset && start <= end
          ? { ...prev, phase: "generating", text: prev.text.slice(0, start - prev.offset) + m.tail }
          : { ...prev, phase: "generating", offset: start, text: m.tail };
      current = { ...current, tokens: m.n ?? 0, percent };
    } else return;
    onPartial(current);
  });
  return (await decode(result.data as { encoding?: string; payload?: string })) as GenerationOutcome;
}

export function metricsText(r: Streaming | GenerationResult | null | undefined) {
  if (!r) return "";
  if ("phase" in r) return `${formatNumber(r.tokens, 0)} tokens · generating`;
  const parts = [
    `${formatNumber(r.generated_tokens, 0)} tokens`,
    `${formatNumber(r.seconds, 2)}s`,
    `${formatNumber(r.tokens_per_second)} tok/s`,
    `${formatNumber(r.load_seconds, 2)}s load`,
    `${formatNumber(r.prompt_tokens, 0)} prompt tokens`,
  ];
  if (r.finish_reason) parts.push(String(r.finish_reason).replaceAll("_", " "));
  return parts.join(" · ");
}

/** The sandboxed frame may not reach the clipboard; say so instead of failing silently. */
export async function copyText(text: string, notify: (message: string) => void) {
  try {
    await navigator.clipboard.writeText(text);
    notify("Copied to clipboard");
  } catch {
    notify("Clipboard unavailable here. Select the text to copy it.");
  }
}

/**
 * fraise-web's Train view: configure a fresh run (architecture preset or a saved run's
 * configuration, form fields or full JSON) or resume a saved training checkpoint (only
 * operational settings change), and check it with fraise's own validation.
 *
 * Training itself runs as an AblationKit job: "Prepare run" writes a checked plan
 * (python/workspace.py:plan → fraise.workbench.training.write_plan), and the MLproject `launch`
 * entrypoint (Runs → Entrypoints) trains it with console, metrics and Cancel. Cancel saves a
 * resumable checkpoint after the current optimizer step, as fraise-web's Stop did.
 */
import { mountReact, Notice, useHost, useHostContext } from "@ablationkit/extension-sdk/react";
import { useEffect, useState } from "react";
import { decode } from "./codec.ts";
import {
  CheckpointOptions,
  type Device,
  formatNumber,
  type Json,
  parseRef,
  type Run,
  run,
  runDecoded,
  useRuns,
} from "./common.tsx";
import { CSS } from "./styles.ts";

interface Field {
  name: string;
  type: "boolean" | "number" | "string" | "object" | "array";
  default: unknown;
}

interface Architecture {
  id: string;
  label: string;
  example: Json;
  fields: Field[];
}

interface Tokenizer {
  id: string;
  label: string;
  vocab_size: number;
}

interface TrainCatalog {
  architectures: Architecture[];
  tokenizers: Tokenizer[];
  sources: string[];
  model_defaults: Json;
  training_defaults: Json;
  operational_settings: string[];
  device: Device;
}

interface Plan {
  run_name: string;
  mode: "fresh" | "resume";
  directory: string;
  parameter: string;
  device: string;
  parameter_count: number;
  resume: string | null;
}

const COMMON_TRAINING = [
  "batch_size",
  "minibatch_size",
  "num_epochs",
  "sequence_length",
  "max_train_tokens",
  "learning_rate",
  "adamw_learning_rate",
  "use_bf16",
  "use_compile",
  "checkpoint_interval_steps",
  "log_interval_steps",
];
const LABELS: Record<string, string> = {
  residual_stream_dim: "Residual dimension",
  head_dim: "Head dimension",
  qk_dim: "Query / key dimension",
  num_heads: "Attention heads",
  swiglu_up_dim: "SwiGLU dimension",
  num_layers: "Layers",
  max_seq_len: "Context length",
  tie_word_embeddings: "Tie word embeddings",
  vocab_size: "Vocabulary size",
  batch_size: "Batch size",
  minibatch_size: "Minibatch size",
  num_epochs: "Epochs",
  sequence_length: "Sequence length",
  max_train_tokens: "Training tokens / epoch",
  learning_rate: "Muon learning rate",
  adamw_learning_rate: "AdamW learning rate",
  use_bf16: "Use bfloat16",
  use_compile: "Compile model",
  checkpoint_interval_steps: "Checkpoint every (steps)",
  log_interval_steps: "Log every (steps)",
};
const labelFor = (name: string) => LABELS[name] ?? name.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
const clone = <T,>(value: T): T => structuredClone(value);
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function FieldInput(props: {
  name: string;
  value: unknown;
  type?: string;
  disabled: boolean;
  onChange: (value: unknown) => void;
}) {
  const { name, value } = props;
  if (typeof value === "boolean" || props.type === "boolean")
    return (
      <label className="fw-field check">
        <input type="checkbox" checked={Boolean(value)} disabled={props.disabled} onChange={(e) => props.onChange(e.target.checked)} />
        {labelFor(name)}
      </label>
    );
  const numeric = typeof value === "number" || props.type === "number" || value == null;
  return (
    <label className="fw-field">
      {labelFor(name)}
      <input
        type={numeric ? "number" : "text"}
        min={numeric ? 0 : undefined}
        step={numeric ? (Number.isInteger(value) && !name.includes("learning_rate") ? 1 : "any") : undefined}
        value={value == null ? "" : String(value)}
        disabled={props.disabled}
        onChange={(e) =>
          props.onChange(numeric ? (e.target.value === "" ? null : Number(e.target.value)) : e.target.value)
        }
      />
    </label>
  );
}

function Train() {
  const host = useHost();
  const ctx = useHostContext();
  const { data: runsData, error: runsError, reload } = useRuns();
  const [catalog, setCatalog] = useState<TrainCatalog>();
  const [catalogError, setCatalogError] = useState<string>();
  const [mode, setMode] = useState<"fresh" | "resume">("fresh");
  const [model, setModel] = useState<Json>({});
  const [training, setTraining] = useState<Json>({});
  const [modelText, setModelText] = useState("");
  const [trainingText, setTrainingText] = useState("");
  const [jsonError, setJsonError] = useState<string>();
  const [name, setName] = useState("run-001");
  const [tokenizer, setTokenizer] = useState("");
  const [preset, setPreset] = useState("");
  const [resume, setResume] = useState("");
  const [freshDraft, setFreshDraft] = useState<{ model: Json; training: Json; name: string; tokenizer: string }>();
  const [busy, setBusy] = useState(false);
  const [checked, setChecked] = useState<number>();
  const [error, setError] = useState<string>();
  const [plan, setPlan] = useState<Plan>();
  const [initialized, setInitialized] = useState(false);

  const runs = runsData?.runs ?? [];
  const resumeRef = parseRef(resume);
  const resumeRun = runs.find((r) => r.id === resumeRef?.run_id);

  const uniqueName = (base: string, existing: Run[]) => {
    const names = new Set(existing.map((r) => r.id));
    let candidate = base;
    for (let i = 2; names.has(candidate); i++) candidate = `${base}-${i}`;
    return candidate;
  };

  const setBoth = (nextModel: Json, nextTraining: Json) => {
    setModel(nextModel);
    setTraining(nextTraining);
    setModelText(pretty(nextModel));
    setTrainingText(pretty(nextTraining));
    setJsonError(undefined);
    setChecked(undefined);
    setPlan(undefined);
  };

  useEffect(() => {
    if (ctx.preview) return;
    runDecoded<TrainCatalog>(host, "train-catalog").then(setCatalog, (e: Error) => setCatalogError(e.message));
  }, [host, ctx.preview]);

  // First load: the default architecture's example and the default training settings.
  useEffect(() => {
    if (!catalog || !runsData || initialized) return;
    setInitialized(true);
    const architecture =
      catalog.architectures.find((a) => a.id === catalog.model_defaults["architecture"]) ?? catalog.architectures[0];
    const first: Json = { ...clone(catalog.model_defaults), ...(architecture ? clone(architecture.example) : {}), architecture: architecture?.id };
    const firstTokenizer = catalog.tokenizers[0];
    if (firstTokenizer) first["vocab_size"] = firstTokenizer.vocab_size;
    setTokenizer(firstTokenizer?.id ?? "");
    setPreset(architecture ? `preset:${architecture.id}` : "");
    setName(uniqueName("run-001", runsData.runs));
    setBoth(first, clone(catalog.training_defaults));
  }, [catalog, runsData, initialized]);

  if (ctx.preview) return <Notice>Preview: Train runs fraise's Python once the build is enabled.</Notice>;
  if (!catalog || !runsData)
    return (
      <div className="fw-root">
        <style>{CSS}</style>
        {catalogError || runsError ? (
          <Notice kind="error">Could not read fraise's catalog: {catalogError ?? runsError}</Notice>
        ) : (
          <Notice>Loading architectures, tokenizers and saved runs…</Notice>
        )}
      </div>
    );

  const resuming = mode === "resume";
  const operational = new Set(catalog.operational_settings);
  const architecture = catalog.architectures.find((a) => a.id === model["architecture"]);

  const applyTokenizer = (next: Json, id: string) => {
    const t = catalog.tokenizers.find((x) => x.id === id);
    return t && !resuming ? { ...next, vocab_size: t.vocab_size } : next;
  };

  const choosePreset = (value: string) => {
    setPreset(value);
    if (value.startsWith("clone:")) {
      const source = runs.find((r) => r.id === value.slice(6));
      if (!source?.config || !source.training) return;
      setName(uniqueName(`${source.id}-copy`, runs));
      setTokenizer(source.tokenizer?.id ?? "");
      setBoth(clone(source.config), clone(source.training));
    } else {
      const a = catalog.architectures.find((x) => x.id === value.slice(7));
      if (a) setBoth(applyTokenizer({ ...clone(a.example), architecture: a.id }, tokenizer), clone(catalog.training_defaults));
    }
  };

  const chooseArchitecture = (id: string) => {
    const a = catalog.architectures.find((x) => x.id === id);
    if (a) setBoth(applyTokenizer({ ...clone(a.example), architecture: a.id }, tokenizer), training);
  };

  const loadResume = (value: string) => {
    setResume(value);
    const ref = parseRef(value);
    const source = runs.find((r) => r.id === ref?.run_id);
    if (!source?.config || !source.training) return;
    setName(source.name || source.id);
    setTokenizer(source.tokenizer?.id ?? "");
    setBoth(clone(source.config), clone(source.training));
  };

  const changeMode = (next: "fresh" | "resume") => {
    if (busy || next === mode) return;
    setMode(next);
    setError(undefined);
    if (next === "resume") {
      setFreshDraft({ model, training, name, tokenizer });
      if (resume) loadResume(resume);
    } else if (freshDraft) {
      setName(freshDraft.name);
      setTokenizer(freshDraft.tokenizer);
      setBoth(freshDraft.model, freshDraft.training);
    }
  };

  const setModelField = (field: string, value: unknown) => setBoth({ ...model, [field]: value }, training);
  const setTrainingField = (field: string, value: unknown) => setBoth(model, { ...training, [field]: value });
  const counts = (training["dataset_token_counts"] as Record<string, number> | undefined) ?? {};
  const setSource = (source: string, value: number) =>
    setBoth(model, { ...training, dataset_token_counts: { ...counts, [source]: value } });
  const sourceTotal = Object.values(counts).reduce((sum, v) => sum + Number(v || 0), 0);
  const budget = Number(training["max_train_tokens"]);

  const readJson = (section: "model" | "training", text: string) => {
    if (section === "model") setModelText(text);
    else setTrainingText(text);
    setChecked(undefined);
    setPlan(undefined);
    try {
      const value = JSON.parse(text) as unknown;
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Use a JSON object.");
      const object = value as Json;
      if (resuming && section === "training" && resumeRun?.training) {
        const saved = resumeRun.training;
        for (const key of new Set([...Object.keys(saved), ...Object.keys(object)]))
          if (!operational.has(key) && JSON.stringify(object[key]) !== JSON.stringify(saved[key]))
            throw new Error(`${key} is fixed when resuming. Only operational settings can change.`);
      }
      if (section === "model") setModel(object);
      else setTraining(object);
      setJsonError(undefined);
    } catch (e) {
      setJsonError(`Invalid ${section} configuration: ${(e as Error).message}`);
    }
  };

  const request = () => {
    if (jsonError) throw new Error(jsonError);
    const runName = name.trim();
    if (!NAME_PATTERN.test(runName))
      throw new Error("Start the run name with a letter or number; use letters, numbers, dots, dashes, and underscores.");
    if (!resuming && runs.some((r) => r.id === runName || r.name === runName))
      throw new Error("That run name already exists. Choose a unique name, or select Resume run.");
    if (resuming && !resumeRef) throw new Error("Choose a training checkpoint to resume.");
    if (!tokenizer) throw new Error("Select a tokenizer.");
    return {
      model_config: model,
      training_config: training,
      tokenizer_id: tokenizer,
      run_name: runName,
      resume: resuming ? resumeRef : null,
    };
  };

  const submit = async (target: "validate" | "plan") => {
    if (busy) return;
    setBusy(true);
    setError(undefined);
    setPlan(undefined);
    try {
      const payload = request();
      const result = await run(host, target, payload as unknown as Json);
      if (target === "validate") {
        const normalized = (await decode(result.data as { encoding?: string; payload?: string })) as {
          model_config: Json;
          training_config: Json;
          parameter_count: number;
        };
        setBoth(normalized.model_config, normalized.training_config);
        setChecked(normalized.parameter_count);
      } else {
        const written = result.data as unknown as Plan;
        setChecked(written.parameter_count);
        setPlan(written);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const modelFields = (architecture?.fields.length ? architecture.fields : Object.keys(model).map((n) => ({ name: n, type: undefined, default: undefined })))
    .filter((f) => f.name !== "architecture")
    .filter((f) => {
      const value = model[f.name] ?? f.default;
      return !(f.type === "object" || f.type === "array" || (value !== null && typeof value === "object"));
    });
  const resumable = runs.filter((r) => r.can_resume);
  const cannotStart = busy || !tokenizer || (resuming && !resumeRun);

  return (
    <div className="fw-root" data-testid="fraise-train">
      <style>{CSS}</style>
      <div className="fw-toolbar" style={{ marginTop: 0, marginBottom: 10 }}>
        <div className="fw-segmented" role="group" aria-label="Training mode">
          {(["fresh", "resume"] as const).map((m) => (
            <button key={m} type="button" className={mode === m ? "active" : ""} aria-pressed={mode === m} disabled={busy} onClick={() => changeMode(m)}>
              {m === "fresh" ? "Fresh run" : "Resume run"}
            </button>
          ))}
        </div>
        <span className="fw-help">
          Device for the job: {catalog.device.resolved.toUpperCase()} · runs are saved in <code>{runsData.models_dir}</code>
        </span>
      </div>
      <div className="fw-layout">
        <div>
          <div className="fw-panel">
            <div className="fw-heading">
              <h2>Run configuration</h2>
              <button type="button" className="fw-link" onClick={reload} disabled={busy}>
                Refresh runs
              </button>
            </div>
            <div className="fw-grid">
              {resuming ? (
                <label className="fw-field wide">
                  Resume from checkpoint
                  <select value={resume} disabled={busy} onChange={(e) => loadResume(e.target.value)}>
                    <option value="">Choose a saved checkpoint</option>
                    <CheckpointOptions runs={resumable} filter={(c) => c.kind === "checkpoint"} />
                  </select>
                </label>
              ) : (
                <label className="fw-field wide">
                  Preset or saved configuration
                  <select value={preset} disabled={busy} onChange={(e) => choosePreset(e.target.value)}>
                    {catalog.architectures.map((a) => (
                      <option key={a.id} value={`preset:${a.id}`}>
                        {a.label} preset
                      </option>
                    ))}
                    <optgroup label="Clone a saved configuration">
                      {runs
                        .filter((r) => r.config && r.training)
                        .map((r) => (
                          <option key={r.id} value={`clone:${r.id}`}>
                            {r.name || r.id}
                          </option>
                        ))}
                    </optgroup>
                  </select>
                </label>
              )}
              <label className="fw-field wide">
                Run name
                <input value={name} maxLength={100} placeholder="run-001" disabled={busy || resuming} onChange={(e) => setName(e.target.value)} />
              </label>
              <label className="fw-field">
                Architecture
                <select value={String(model["architecture"] ?? "")} disabled={busy || resuming} onChange={(e) => chooseArchitecture(e.target.value)}>
                  {catalog.architectures.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="fw-field">
                Tokenizer
                <select
                  value={tokenizer}
                  disabled={busy || resuming}
                  onChange={(e) => {
                    setTokenizer(e.target.value);
                    setBoth(applyTokenizer(model, e.target.value), training);
                  }}
                >
                  <option value="">Select a tokenizer</option>
                  {catalog.tokenizers.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.label} · {formatNumber(t.vocab_size, 0)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p className="fw-help">
              {catalog.tokenizers.length
                ? "The tokenizer sets the model’s vocabulary size and is saved with the run."
                : "No tokenizer is available. Create one with fraise’s create-tokenizer command, then refresh."}
            </p>
            {resuming ? (
              <p className="fw-help">
                {resumable.length
                  ? "Resuming preserves the saved model, tokenizer, and training identity. Only operational settings can change."
                  : "No resumable checkpoints are available. Start a fresh run or migrate an older saved run first."}
              </p>
            ) : null}
          </div>

          <div className="fw-panel">
            <div className="fw-heading">
              <h2>Model configuration</h2>
              <span className="fw-badge">{checked == null ? "Not yet checked" : `${formatNumber(checked, 0)} parameters`}</span>
            </div>
            <div className="fw-grid">
              {modelFields.map((f) => (
                <FieldInput
                  key={f.name}
                  name={f.name}
                  value={model[f.name] ?? f.default}
                  type={f.type}
                  disabled={busy || resuming || f.name === "vocab_size"}
                  onChange={(v) => setModelField(f.name, v)}
                />
              ))}
            </div>
            <details className="fw-details">
              <summary>Full model configuration</summary>
              <textarea
                className="fw-json"
                aria-label="Model configuration JSON"
                spellCheck={false}
                disabled={busy || resuming}
                value={modelText}
                onChange={(e) => readJson("model", e.target.value)}
              />
            </details>
          </div>

          <div className="fw-panel">
            <div className="fw-heading">
              <h2>Training settings</h2>
            </div>
            <div className="fw-grid">
              {COMMON_TRAINING.filter((n) => n in training).map((n) => (
                <FieldInput
                  key={n}
                  name={n}
                  value={training[n]}
                  disabled={busy || (resuming && !operational.has(n))}
                  onChange={(v) => setTrainingField(n, v)}
                />
              ))}
            </div>
            <h3 style={{ fontSize: 13, margin: "14px 0 6px" }}>Dataset token budgets</h3>
            <div className="fw-grid">
              {[...new Set([...catalog.sources, ...Object.keys(counts)])].map((source) => (
                <label key={source} className="fw-field">
                  {labelFor(source)}
                  <input
                    type="number"
                    min={0}
                    step={1}
                    disabled={busy || resuming}
                    value={counts[source] ?? 0}
                    onChange={(e) => setSource(source, e.target.value === "" ? 0 : Number(e.target.value))}
                  />
                </label>
              ))}
            </div>
            <div className={`fw-total${sourceTotal === budget ? "" : " invalid"}`}>
              <span>Source total: {formatNumber(sourceTotal, 0)}</span>
              <span>{sourceTotal === budget ? "Matches training token budget ✓" : `Must equal ${formatNumber(budget, 0)} tokens`}</span>
            </div>
            <details className="fw-details">
              <summary>Full training configuration</summary>
              <p className="fw-help">
                Common controls and JSON edit the same configuration. Use JSON for optimizer, validation, packing, and other
                advanced settings.
                {resuming ? ` Editable operational settings: ${catalog.operational_settings.join(", ")}.` : ""}
              </p>
              <textarea
                className="fw-json"
                aria-label="Training configuration JSON"
                spellCheck={false}
                disabled={busy || (resuming && !resumeRun)}
                value={trainingText}
                onChange={(e) => readJson("training", e.target.value)}
              />
            </details>
            {jsonError ? <p className="fw-error" role="alert">{jsonError}</p> : null}
            {checked != null && !plan && !error ? (
              <p className="fw-ok" role="status">Configuration is ready · {formatNumber(checked, 0)} parameters</p>
            ) : null}
            {error ? <p className="fw-error" role="alert">{error}</p> : null}
            <div className="fw-toolbar">
              <span className="fw-help">Prepare writes a checked plan; AblationKit runs it as a job.</span>
              <div className="fw-buttons">
                <button type="button" className="fw-button" disabled={cannotStart} onClick={() => void submit("validate")}>
                  Check configuration
                </button>
                <button type="button" className="fw-button primary" disabled={cannotStart} onClick={() => void submit("plan")}>
                  {busy ? "Preparing…" : resuming ? "Prepare resume" : "Prepare run"}
                </button>
              </div>
            </div>
          </div>
        </div>

        <aside className="fw-side">
          {plan ? (
            <div className="fw-plan" data-testid="fraise-train-plan">
              <h2 style={{ margin: 0, fontSize: 14 }}>
                {plan.mode === "resume" ? "Resume" : "Run"} {plan.run_name} is ready
              </h2>
              <p className="fw-help" style={{ margin: 0 }}>
                {formatNumber(plan.parameter_count, 0)} parameters on {plan.device.toUpperCase()}
                {plan.resume ? ` · from ${plan.resume.split("/").slice(-2).join("/")}` : ""}
              </p>
              <ol className="fw-help">
                <li>
                  Open <strong>Runs → More → Entrypoints</strong>.
                </li>
                <li>
                  <strong>Run…</strong> the <code>launch</code> entrypoint with <code>plan</code>:
                </li>
              </ol>
              <code>{plan.parameter}</code>
              <ol className="fw-help" start={3}>
                <li>
                  <strong>Run as a job.</strong> The job page shows the console; the run page shows loss, throughput and
                  validation.
                </li>
                <li>
                  <strong>Cancel</strong> the job to save a resumable checkpoint after the current optimizer step.
                </li>
              </ol>
              <div className="fw-buttons">
                <button type="button" className="fw-button primary" onClick={() => host.navigate(`/p/${ctx.project_id}/runs/entrypoints`)}>
                  Open Entrypoints
                </button>
                <button type="button" className="fw-button" onClick={() => host.navigate(`/p/${ctx.project_id}/runs/jobs`)}>
                  Open Jobs
                </button>
              </div>
              <p className="fw-help" style={{ margin: 0 }}>
                Free the GPU first if Autocomplete or Compare loaded a checkpoint (their “Release device memory”).
              </p>
            </div>
          ) : (
            <div className="fw-panel">
              <h2 style={{ margin: "0 0 8px", fontSize: 14 }}>Training as a job</h2>
              <p className="fw-help">
                Prepare a checked run here, then launch it from Runs → Entrypoints (<code>launch</code>). AblationKit
                supervises it: console logs, loss and validation curves on its run, and Cancel, which saves a resumable
                checkpoint after the current optimizer step.
              </p>
              <p className="fw-help">
                Resumable runs: {resumable.length} · saved runs: {runs.length}
                {runs.some((r) => r.error) ? ` · ${runs.filter((r) => r.error).length} unreadable` : ""}
              </p>
              {resuming && resumeRun ? (
                <p className="fw-help">
                  Resuming {resumeRun.name} at {formatNumber(resumeRun.global_step, 0)} steps,{" "}
                  {formatNumber(resumeRun.tokens_seen, 0)} tokens seen.
                </p>
              ) : null}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

void mountReact(Train);

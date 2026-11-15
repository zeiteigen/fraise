import type { RunSummary } from "@ablationkit/extension-sdk";
import { mountReact, Notice, Table, useHost, useHostContext } from "@ablationkit/extension-sdk/react";
import { useEffect, useState } from "react";
import {
  AXES,
  type Axis,
  type Comparison,
  compare,
  curve,
  formatX,
  type Point,
  type RunSeries,
  xAt,
} from "./axes.ts";
import { CHART_CSS, Chart } from "./Chart.tsx";

const BPB = "validation/bits_per_byte";
const MAX_SERIES = 8;

async function loadRun(
  host: ReturnType<typeof useHost>,
  runId: string,
): Promise<RunSeries & { summary: RunSummary }> {
  const history = (key: string) =>
    host
      .call<{ history: Point[] }>("runs.metric_history", { run_id: runId, key })
      .then((h) => h.history);
  const [summary, bpb, tokens] = await Promise.all([
    host.call<RunSummary>("runs.get", { run_id: runId }),
    history(BPB),
    history("tokens_seen"),
  ]);
  const batch = Number(summary.params["batch_size"]);
  const length = Number(summary.params["sequence_length"]);
  return {
    summary,
    runId,
    name: summary.run_name ?? runId,
    bpb,
    tokens,
    positionsPerStep: batch > 0 && length > 0 ? batch * length : undefined,
  };
}

/** Default reference: the base 8k transformer if one is selected, else the first run. */
function defaultReference(runs: (RunSeries & { summary: RunSummary })[]): string | undefined {
  const base = runs.find(
    (r) => r.summary.params["architecture"] === "transformer" && r.summary.params["vocab_size"] === "8192",
  );
  return (base ?? runs[0])?.runId;
}

function readingText(c: Comparison): string {
  const r = c.reading;
  switch (r.kind) {
    case "exact":
      return `${r.y} (exact, step ${r.between[0]})`;
    case "interpolated":
      return `${r.y.toFixed(4)} (interpolated, steps ${r.between[0]}–${r.between[1]})`;
    case "out_of_range":
      return `out of the reference's range (${formatX(r.min)}–${formatX(r.max)})`;
    case "none":
      return "—";
  }
}

function deltaText(c: Comparison): string {
  if (typeof c.final === "string") return "—";
  if (c.reading.kind !== "exact" && c.reading.kind !== "interpolated") return "—";
  const d = c.final.y - c.reading.y;
  return `${d > 0 ? "+" : ""}${d.toFixed(4)} ${d > 0 ? "worse" : d < 0 ? "better" : ""}`;
}

function Compare() {
  const host = useHost();
  const ctx = useHostContext();
  const [runs, setRuns] = useState<(RunSeries & { summary: RunSummary })[]>([]);
  const [error, setError] = useState<string>();
  const [axis, setAxis] = useState<Axis>("tokens");
  const [referenceId, setReferenceId] = useState<string>();
  const [flops, setFlops] = useState<Record<string, string>>({});

  useEffect(() => {
    if (ctx.preview) return;
    let live = true;
    Promise.all((ctx.run_ids ?? []).map((id) => loadRun(host, id))).then(
      (loaded) => {
        if (!live) return;
        setRuns(loaded);
        setReferenceId((cur) =>
          cur && loaded.some((r) => r.runId === cur) ? cur : defaultReference(loaded),
        );
      },
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [host, ctx.run_ids, ctx.preview]);

  if (ctx.preview)
    return <Notice>Preview: run data is not available until the build is approved.</Notice>;
  if (error) return <Notice kind="error">Could not read the runs: {error}</Notice>;
  if ((ctx.run_ids ?? []).length < 2)
    return <Notice>Select at least two runs (a reference and a candidate) to compare.</Notice>;
  if (runs.length === 0) return <Notice>Loading runs…</Notice>;

  const series = runs.map((r) => {
    const value = Number(flops[r.runId]);
    return { ...r, flopsPerToken: value > 0 ? value : undefined };
  });
  const reference = series.find((r) => r.runId === referenceId) ?? series[0];
  const meta = AXES.find((a) => a.id === axis) ?? AXES[0];
  if (!reference || !meta) return null;
  const candidates = series.filter((r) => r.runId !== reference.runId);
  const comparisons = candidates.map((c) => compare(reference, c, axis));
  const chartSeries = series.slice(0, MAX_SERIES).map((r, i) => ({
    id: r.runId,
    name: r.name,
    slot: i + 1,
    points: curve(r, axis),
  }));
  const missingBpb = series.filter((r) => r.bpb.length === 0).map((r) => r.name);

  return (
    <div data-testid="compute-axes">
      <style>{CHART_CSS}</style>
      <Notice>
        Each candidate's final validation BPB against the reference's validation curve read at
        the same compute. Reference values between validation points are linearly interpolated
        and never extrapolated. They come from partway through the reference's single run, before
        its learning rate annealed, so a reference trained to that budget would likely score
        better (pilot report §5.4). Seed variance is unmeasured.
      </Notice>

      <div className="ca-controls">
        <span>Axis:</span>
        {AXES.map((a) => (
          <label key={a.id}>
            <input type="radio" name="axis" checked={axis === a.id} onChange={() => setAxis(a.id)} />{" "}
            {a.label}
          </label>
        ))}
      </div>
      <div className="ca-controls">
        <label>
          Reference run:{" "}
          <select value={reference.runId} onChange={(e) => setReferenceId(e.target.value)}>
            {series.map((r) => (
              <option key={r.runId} value={r.runId}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {axis === "flops" ? (
        <div className="ca-controls ca-flops">
          <span>FLOPs per base token (your estimate; no run records it):</span>
          {series.map((r) => (
            <label key={r.runId}>
              {r.name}{" "}
              <input
                inputMode="decimal"
                placeholder="FLOPs, e.g. 5e8"
                value={flops[r.runId] ?? ""}
                onChange={(e) => setFlops({ ...flops, [r.runId]: e.target.value })}
              />
            </label>
          ))}
        </div>
      ) : null}

      {missingBpb.length ? (
        <Notice kind="error">No {BPB} history: {missingBpb.join(", ")}.</Notice>
      ) : null}
      {series.length > MAX_SERIES ? (
        <Notice>
          The chart shows the first {MAX_SERIES} runs; the tables below include all {series.length}.
        </Notice>
      ) : null}

      <Chart series={chartSeries} xLabel={`${meta.label.toLowerCase()} (${meta.unit})`} />

      <h3>Against {reference.name}</h3>
      <Table
        columns={[
          "candidate",
          "final step",
          `at (${meta.unit})`,
          "candidate BPB",
          "reference BPB there",
          "Δ candidate − reference",
        ]}
        rows={comparisons.map((c) => [
          c.candidate.name,
          typeof c.final === "string" ? "—" : c.final.step,
          typeof c.final === "string" ? c.final : formatX(c.final.x),
          typeof c.final === "string" ? "—" : c.final.y,
          readingText(c),
          deltaText(c),
        ])}
      />

      <h3>Final validation point of each run on every axis</h3>
      <Table
        columns={["run", "step", BPB, ...AXES.map((a) => a.unit)]}
        rows={series.map((r) => {
          const last = [...r.bpb].sort((a, b) => a.step - b.step).at(-1);
          if (!last) return [r.name, "—", "—", ...AXES.map(() => "—")];
          return [
            r.name,
            last.step,
            last.value,
            ...AXES.map((a) => {
              const x = xAt(r, a.id, last.step);
              return typeof x === "number" ? formatX(x) : x;
            }),
          ];
        })}
      />
    </div>
  );
}

void mountReact(Compare);

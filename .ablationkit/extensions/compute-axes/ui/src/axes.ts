/**
 * Compute axes for the pilot comparison (pilot report §5.4), computed only from the runs' own
 * recorded data:
 *
 * - tokens:    `tokens_seen` (base tokens) logged at the validation step;
 * - positions: step × batch_size × sequence_length (training global positions; params);
 * - seconds:   time of the training point at the validation step minus the time of the first
 *              logged training point (excludes data preparation, compilation and the first log
 *              interval);
 * - flops:     tokens × a FLOPs-per-base-token figure the researcher types in (an estimate; no run
 *              records it).
 *
 * The reference curve is interpolated linearly between its own validation points and never
 * extrapolated.
 */

export type Axis = "tokens" | "positions" | "seconds" | "flops";

export const AXES: { id: Axis; label: string; unit: string }[] = [
  { id: "tokens", label: "Equal tokens of text", unit: "base tokens" },
  { id: "flops", label: "Equal training FLOPs (your estimate)", unit: "FLOPs" },
  { id: "seconds", label: "Equal wall-clock", unit: "s since first logged step" },
  { id: "positions", label: "Equal training global positions", unit: "positions" },
];

export interface Point {
  step: number;
  value: number;
  timestamp: number;
}

export interface RunSeries {
  runId: string;
  name: string;
  /** validation/bits_per_byte history. */
  bpb: Point[];
  /** tokens_seen history. */
  tokens: Point[];
  /** batch_size × sequence_length, when both params are recorded. */
  positionsPerStep?: number | undefined;
  /** Researcher-entered FLOPs per base token. */
  flopsPerToken?: number | undefined;
}

export interface CurvePoint {
  step: number;
  x: number;
  y: number;
}

const byStep = (a: Point, b: Point) => a.step - b.step;

/** The x value of `step` on `axis`, or why there is none. */
export function xAt(run: RunSeries, axis: Axis, step: number): number | string {
  const atStep = run.tokens.find((p) => p.step === step);
  switch (axis) {
    case "tokens":
      return atStep ? atStep.value : "no tokens_seen at this step";
    case "flops":
      if (!run.flopsPerToken) return "no FLOPs-per-token estimate entered";
      return atStep ? atStep.value * run.flopsPerToken : "no tokens_seen at this step";
    case "positions":
      return run.positionsPerStep
        ? step * run.positionsPerStep
        : "no batch_size / sequence_length params";
    case "seconds": {
      const first = [...run.tokens].sort(byStep)[0];
      if (!first || !atStep) return "no training timestamps at this step";
      return (atStep.timestamp - first.timestamp) / 1000;
    }
  }
}

/** The run's validation BPB curve on `axis` (points without an x value are left out). */
export function curve(run: RunSeries, axis: Axis): CurvePoint[] {
  return [...run.bpb].sort(byStep).flatMap((p) => {
    const x = xAt(run, axis, p.step);
    return typeof x === "number" ? [{ step: p.step, x, y: p.value }] : [];
  });
}

export type Reading =
  | { kind: "exact" | "interpolated"; y: number; between: [number, number] }
  | { kind: "out_of_range"; min: number; max: number }
  | { kind: "none" };

/** The reference curve's value at x: exact, linearly interpolated, or out of its range. */
export function readAt(points: readonly CurvePoint[], x: number): Reading {
  if (points.length === 0) return { kind: "none" };
  const first = points[0] as CurvePoint;
  const last = points[points.length - 1] as CurvePoint;
  if (x < first.x || x > last.x) return { kind: "out_of_range", min: first.x, max: last.x };
  for (let i = 0; i < points.length; i++) {
    const p = points[i] as CurvePoint;
    if (p.x === x) return { kind: "exact", y: p.y, between: [p.step, p.step] };
    const q = points[i + 1];
    if (q && p.x < x && x < q.x) {
      const t = (x - p.x) / (q.x - p.x);
      return { kind: "interpolated", y: p.y + t * (q.y - p.y), between: [p.step, q.step] };
    }
  }
  return { kind: "none" };
}

export interface Comparison {
  candidate: RunSeries;
  /** The candidate's final validation point on this axis, or why there is none. */
  final: CurvePoint | string;
  reading: Reading;
}

/** Each candidate's final validation BPB against the reference read at the same x. */
export function compare(reference: RunSeries, candidate: RunSeries, axis: Axis): Comparison {
  const last = [...candidate.bpb].sort(byStep).at(-1);
  if (!last) return { candidate, final: "no validation/bits_per_byte", reading: { kind: "none" } };
  const x = xAt(candidate, axis, last.step);
  if (typeof x !== "number") return { candidate, final: x, reading: { kind: "none" } };
  return {
    candidate,
    final: { step: last.step, x, y: last.value },
    reading: readAt(curve(reference, axis), x),
  };
}

/** Compact number formatting for axis ticks and tables (no rounding of BPB). */
export function formatX(x: number): string {
  const abs = Math.abs(x);
  if (abs >= 1e15) return `${(x / 1e15).toPrecision(4)}P`;
  if (abs >= 1e12) return `${(x / 1e12).toPrecision(4)}T`;
  if (abs >= 1e9) return `${(x / 1e9).toPrecision(4)}G`;
  if (abs >= 1e6) return `${(x / 1e6).toPrecision(4)}M`;
  if (abs >= 1e3) return `${(x / 1e3).toPrecision(4)}k`;
  return x.toPrecision(4);
}

/** About five round ticks covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) ?? 10) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

import { useState } from "react";
import { type CurvePoint, formatX, ticks } from "./axes.ts";

export interface ChartSeries {
  id: string;
  name: string;
  /** Categorical slot, 1-based, fixed per run (never by rank). */
  slot: number;
  points: CurvePoint[];
}

const W = 680;
const H = 320;
const M = { top: 16, right: 140, bottom: 44, left: 56 };

/** Validation BPB against one compute axis: 2px lines, 8px markers, hover/focus readout. */
export function Chart(props: { series: ChartSeries[]; xLabel: string }) {
  const [hover, setHover] = useState<{ s: ChartSeries; p: CurvePoint } | null>(null);
  const all = props.series.flatMap((s) => s.points);
  if (all.length === 0) return null;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xMax = Math.max(...xs);
  const yLo = Math.min(...ys);
  const yHi = Math.max(...ys);
  const pad = (yHi - yLo) * 0.06 || 0.01;
  const y0 = yLo - pad;
  const y1 = yHi + pad;
  const sx = (x: number) => M.left + (x / (xMax || 1)) * (W - M.left - M.right);
  const sy = (y: number) => M.top + ((y1 - y) / (y1 - y0)) * (H - M.top - M.bottom);
  const direct = props.series.length <= 4;

  return (
    <figure className="viz-root" style={{ margin: 0, position: "relative" }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label={`Validation bits per byte against ${props.xLabel}`}
      >
        {ticks(y0, y1).map((t) => (
          <g key={`y${t}`}>
            <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} className="viz-grid" />
            <text x={M.left - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="viz-tick">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {ticks(0, xMax).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - M.bottom + 18} textAnchor="middle" className="viz-tick">
            {formatX(t)}
          </text>
        ))}
        <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} className="viz-axis" />
        <text x={(M.left + W - M.right) / 2} y={H - 6} textAnchor="middle" className="viz-label">
          {props.xLabel}
        </text>
        <text x={14} y={M.top + 4} className="viz-label" transform={`rotate(-90 14 ${M.top + 4})`} textAnchor="end">
          validation BPB (lower is better)
        </text>
        {props.series.map((s) => {
          const color = `var(--series-${s.slot})`;
          const path = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join("");
          const end = s.points.at(-1);
          return (
            <g key={s.id}>
              <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
              {s.points.map((p) => (
                <g key={p.step}>
                  <circle cx={sx(p.x)} cy={sy(p.y)} r={4} fill={color} className="viz-marker" />
                  <circle
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={10}
                    fill="transparent"
                    tabIndex={0}
                    aria-label={`${s.name}, step ${p.step}: ${p.y}`}
                    onPointerEnter={() => setHover({ s, p })}
                    onPointerLeave={() => setHover(null)}
                    onFocus={() => setHover({ s, p })}
                    onBlur={() => setHover(null)}
                  />
                </g>
              ))}
              {direct && end ? (
                <text x={sx(end.x) + 8} y={sy(end.y)} dy="0.32em" className="viz-direct">
                  {s.name}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      {hover ? (
        <div
          className="viz-tooltip"
          style={{ left: `${(sx(hover.p.x) / W) * 100}%`, top: `${(sy(hover.p.y) / H) * 100}%` }}
        >
          <strong>{hover.s.name}</strong>
          <div>step {hover.p.step}</div>
          <div>
            {props.xLabel}: {formatX(hover.p.x)}
          </div>
          <div>validation BPB: {hover.p.y}</div>
        </div>
      ) : null}
      <figcaption className="viz-legend">
        {props.series.map((s) => (
          <span key={s.id}>
            <i style={{ background: `var(--series-${s.slot})` }} />
            {s.name}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

/** Light and dark steps of the validated reference palette, plus chart neutrals. */
export const CHART_CSS = `
.viz-root{--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;--series-6:#008300;--series-7:#6250d6;--series-8:#e34948;--viz-grid:#e1e0d9;--viz-axis:#c3c2b7;--viz-muted:#898781;--viz-surface:#fcfcfb;--viz-text:#0b0b0b;--viz-text-2:#52514e}
:root[data-theme="dark"] .viz-root{--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;--viz-grid:#2c2c2a;--viz-axis:#383835;--viz-muted:#898781;--viz-surface:#1a1a19;--viz-text:#f0efec;--viz-text-2:#c3c2b7}
.viz-grid{stroke:var(--viz-grid);stroke-width:1}
.viz-axis{stroke:var(--viz-axis);stroke-width:1}
.viz-tick{fill:var(--viz-muted);font-size:11px;font-variant-numeric:tabular-nums}
.viz-label{fill:var(--viz-text-2);font-size:12px}
.viz-direct{fill:var(--viz-text);font-size:12px}
.viz-marker{stroke:var(--viz-surface);stroke-width:2}
.viz-tooltip{position:absolute;transform:translate(12px,-50%);pointer-events:none;background:var(--viz-surface);color:var(--viz-text);border:1px solid var(--viz-grid);border-radius:6px;padding:6px 8px;font-size:12px;white-space:nowrap;font-variant-numeric:tabular-nums}
.viz-legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--viz-text-2);margin-top:4px}
.viz-legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.ca-controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:8px 0}
.ca-controls label{font-size:13px}
.ca-flops input{width:9em}
`;

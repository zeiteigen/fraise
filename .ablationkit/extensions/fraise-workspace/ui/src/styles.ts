/** fraise-web's workspace styles (static/styles.css), trimmed and adapted for light and dark hosts. */
export const CSS = `
.fw-root{--fw-line:#d8dee3;--fw-muted:#64707c;--fw-accent:#315f85;--fw-card:#ffffff;--fw-soft:#f8fafb;--fw-ink:#1d252c;--fw-ok:#2f7a4b;--fw-bad:#b3261e;--fw-mono:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--fw-ink);font-size:13px}
:root[data-theme="dark"] .fw-root{--fw-line:#3a4148;--fw-muted:#9aa5af;--fw-accent:#7fb0da;--fw-card:#1b1f23;--fw-soft:#22272c;--fw-ink:#e8ecef;--fw-ok:#6fc58e;--fw-bad:#f2a49d}
.fw-root code,.fw-mono{font-family:var(--fw-mono)}
.fw-root *{box-sizing:border-box}
.fw-panel{min-width:0;padding:14px;margin-bottom:14px;border:1px solid var(--fw-line);border-radius:5px;background:var(--fw-card)}
.fw-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 0 8px}
.fw-heading h2,.fw-heading h3{margin:0;font-size:15px}
.fw-help{color:var(--fw-muted);font-size:11px;line-height:1.6;margin:4px 0}
.fw-badge{padding:2px 6px;border:1px solid var(--fw-line);border-radius:3px;color:var(--fw-muted);font-size:10px;white-space:nowrap}
.fw-badge.running{color:var(--fw-accent);border-color:var(--fw-accent)}
.fw-badge.success{color:var(--fw-ok);border-color:var(--fw-ok)}
.fw-badge.failed{color:var(--fw-bad);border-color:var(--fw-bad)}
.fw-error{color:var(--fw-bad);font-size:12px;margin:6px 0;white-space:pre-wrap}
.fw-ok{color:var(--fw-ok);font-size:12px;margin:6px 0}
.fw-root textarea,.fw-root input,.fw-root select{padding:4px 6px;border:1px solid var(--fw-line);border-radius:4px;background:var(--fw-card);color:var(--fw-ink);font:inherit}
.fw-root textarea:disabled,.fw-root input:disabled,.fw-root select:disabled{opacity:.6}
.fw-document{width:100%;min-height:260px;padding:10px;font:13px/1.6 var(--fw-mono);resize:vertical}
.fw-prompt{width:100%;min-height:90px;padding:8px;font:13px/1.6 var(--fw-mono);resize:vertical}
.fw-json{width:100%;min-height:220px;padding:8px;font:12px/1.5 var(--fw-mono)}
.fw-output{min-height:80px;margin:0;padding:10px;border:1px solid var(--fw-line);border-radius:4px;background:var(--fw-soft);font:13px/1.6 var(--fw-mono);white-space:pre-wrap;overflow-wrap:anywhere}
.fw-status{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;color:var(--fw-muted);font-size:11px;margin:4px 0 8px}
.fw-toolbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin-top:10px}
.fw-buttons{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.fw-button{padding:5px 12px;border:1px solid var(--fw-line);border-radius:4px;background:var(--fw-card);color:var(--fw-ink);font-size:12px;cursor:pointer}
.fw-button.primary{border-color:var(--fw-accent);background:var(--fw-accent);color:var(--fw-card)}
.fw-button:disabled{opacity:.5;cursor:default}
.fw-link{padding:2px 0;border:0;background:none;color:var(--fw-accent);font-size:12px;cursor:pointer}
.fw-link:disabled{color:var(--fw-muted);cursor:default}
.fw-metrics{color:var(--fw-muted);font:11px/1.6 var(--fw-mono)}
.fw-layout{display:grid;gap:14px;grid-template-columns:minmax(0,1fr) 280px;align-items:start}
.fw-layout.wide-side{grid-template-columns:200px minmax(0,1fr) 280px}
@media (max-width:900px){.fw-layout,.fw-layout.wide-side{grid-template-columns:minmax(0,1fr)}}
.fw-side h2{margin:0 0 8px;font-size:14px}
.fw-info{display:grid;gap:6px;margin:0}
.fw-info div{display:grid;grid-template-columns:1fr auto;gap:8px;border-bottom:1px solid var(--fw-line);padding-bottom:4px}
.fw-info dt{color:var(--fw-muted);font-size:11px}
.fw-info dd{margin:0;font:12px var(--fw-mono);text-align:right;overflow-wrap:anywhere}
.fw-details{margin-top:10px}
.fw-details summary{cursor:pointer;font-size:12px;color:var(--fw-accent)}
.fw-details pre{max-height:320px;overflow:auto;font:11px/1.5 var(--fw-mono);background:var(--fw-soft);padding:8px;border-radius:4px}
.fw-settings summary{cursor:pointer;font-size:12px;margin:8px 0}
.fw-settings-grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}
.fw-field{display:grid;gap:3px;font-size:11px;color:var(--fw-muted)}
.fw-field.check{display:flex;align-items:center;gap:6px;color:var(--fw-ink);font-size:12px}
.fw-grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.fw-grid .wide{grid-column:1/-1}
.fw-saved{display:grid;gap:4px}
.fw-saved button{display:grid;gap:2px;padding:6px 8px;border:1px solid var(--fw-line);border-radius:4px;background:var(--fw-soft);color:var(--fw-ink);text-align:left;cursor:pointer}
.fw-saved button.active{border-color:var(--fw-accent)}
.fw-saved small{color:var(--fw-muted);font-size:10px}
.fw-runs{display:grid;gap:8px}
.fw-run{padding:10px;border:1px solid var(--fw-line);border-radius:4px;background:var(--fw-soft)}
.fw-run-heading{display:flex;justify-content:space-between;gap:10px}
.fw-run-heading small{display:block;color:var(--fw-muted);font-size:11px}
.fw-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.fw-chip{display:flex;align-items:center;gap:4px;padding:2px 8px;border:1px solid var(--fw-line);border-radius:12px;background:var(--fw-card);font-size:11px;cursor:pointer}
.fw-results{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));align-items:start}
.fw-card{display:grid;gap:8px;padding:12px;border:1px solid var(--fw-line);border-radius:5px;background:var(--fw-card)}
.fw-card.streaming{border-color:var(--fw-accent)}
.fw-card h3{margin:0;font-size:13px}
.fw-card small{color:var(--fw-muted);font-size:11px}
.fw-segmented{display:inline-flex;border:1px solid var(--fw-line);border-radius:4px;overflow:hidden}
.fw-segmented button{padding:5px 12px;border:0;background:var(--fw-card);color:var(--fw-ink);font-size:12px;cursor:pointer}
.fw-segmented button.active{background:var(--fw-accent);color:var(--fw-card)}
.fw-total{display:flex;justify-content:space-between;gap:10px;margin-top:8px;font-size:12px;color:var(--fw-ok)}
.fw-total.invalid{color:var(--fw-bad)}
.fw-plan{display:grid;gap:8px;padding:12px;border:1px solid var(--fw-ok);border-radius:5px;background:var(--fw-soft)}
.fw-plan code{padding:6px 8px;border-radius:4px;background:var(--fw-card);overflow-wrap:anywhere}
.fw-plan ol{margin:0;padding-left:18px;line-height:1.7}
.fw-progress{height:4px;border-radius:2px;background:var(--fw-line);overflow:hidden}
.fw-progress div{height:100%;background:var(--fw-accent);transition:width .2s}
`;

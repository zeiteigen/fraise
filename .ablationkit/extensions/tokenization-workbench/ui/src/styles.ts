/** fraise-web's tokens view styles (static/styles.css, tok-*), adapted for light and dark hosts. */
export const CSS = `
.tw-root{--tw-line:#d8dee3;--tw-muted:#64707c;--tw-accent:#315f85;--tw-card:#ffffff;--tw-soft:#f8fafb;--tw-ink:#1d252c;--tw-mono:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--tw-ink)}
:root[data-theme="dark"] .tw-root{--tw-line:#3a4148;--tw-muted:#9aa5af;--tw-accent:#7fb0da;--tw-card:#1b1f23;--tw-soft:#22272c;--tw-ink:#e8ecef}
.tw-root code{font-family:var(--tw-mono)}
.tw-panel,.tw-card{min-width:0;padding:14px;margin-bottom:14px;border:1px solid var(--tw-line);border-radius:5px;background:var(--tw-card)}
.tw-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:6px 0}
.tw-heading h2{margin:0;font-size:15px}
.tw-badge{padding:2px 6px;border:1px solid var(--tw-line);border-radius:3px;color:var(--tw-muted);font-size:10px;white-space:nowrap}
.tw-text{width:100%;min-height:90px;box-sizing:border-box;padding:8px;border:1px solid var(--tw-line);border-radius:4px;background:var(--tw-card);color:var(--tw-ink);font:13px/1.6 var(--tw-mono)}
.tw-check{font-size:12px}
.tw-configs{display:grid;gap:8px}
.tw-config{display:flex;flex-wrap:wrap;align-items:flex-end;gap:10px;padding:10px;border:1px solid var(--tw-line);border-radius:4px;background:var(--tw-soft)}
.tw-config label{display:grid;gap:3px;min-width:150px;font-size:11px;color:var(--tw-muted)}
.tw-config select,.tw-config input{max-width:320px;padding:3px 5px;border:1px solid var(--tw-line);border-radius:4px;background:var(--tw-card);color:var(--tw-ink)}
.tw-link{padding:4px 0;border:0;background:none;color:var(--tw-accent);font-size:12px;cursor:pointer}
.tw-link:disabled{color:var(--tw-muted);cursor:default}
.tw-footer{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:10px}
.tw-help{color:var(--tw-muted);font-size:11px;line-height:1.6;margin:4px 0}
.tw-error{color:#b3261e}
.tw-notice{padding:6px 10px;border-left:3px solid var(--tw-accent);font-size:12px}
.tw-results{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));align-items:start}
.tw-metrics{color:var(--tw-muted);font:10px/1.65 var(--tw-mono);margin:4px 0 10px}
.tw-section{margin-top:14px;padding-top:10px;border-top:1px solid var(--tw-line)}
.tw-section>h3,.tw-section>summary{margin:0 0 4px;font-size:13px}
.tw-strip{display:flex;flex-wrap:wrap;align-items:center;gap:3px;margin-top:8px}
.tw-strip-label{width:96px;flex-shrink:0;color:var(--tw-muted);font-size:11px}
.tw-chip{padding:2px 5px;border-radius:3px;font:12px/1.5 var(--tw-mono);white-space:pre;color:#1d252c}
.tw-chip.segment{border:1px solid rgba(0,0,0,.28);font-weight:600;cursor:pointer}
.tw-chip.base{border:1px solid rgba(0,0,0,.08);opacity:.85}
.tw-chip.selected,.tw-cell.selected{outline:2px solid var(--tw-accent);outline-offset:1px}
.c0{background:#fde9b8}.c1{background:#cfe2f7}.c2{background:#d3efd9}.c3{background:#f8d6d3}.c4{background:#e2dcf6}.c5{background:#fbdcc0}
.tw-icicle-scroll{overflow-x:auto;padding:4px 2px 8px}
.tw-icicle{display:grid;gap:3px;width:max-content}
.tw-level-label{grid-column:1;align-self:center;padding-right:8px;color:var(--tw-muted);font:11px/1.4 var(--tw-mono);white-space:nowrap}
.tw-cell{display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:26px;padding:2px 4px;border:1px solid rgba(0,0,0,.3);border-radius:3px;font:12px/1.35 var(--tw-mono);color:#1d252c;cursor:pointer}
.tw-cell small{color:#5c6770;font-size:9px}
.tw-cell-text{white-space:pre}
.tw-cell.carried{border-style:dashed;opacity:.55;background-image:repeating-linear-gradient(135deg,transparent 0 5px,rgba(255,255,255,.55) 5px 8px)}
.tw-cell.focused{box-shadow:0 0 0 2px var(--tw-ink);opacity:1}
.tw-table-scroll{max-width:100%;overflow-x:auto;margin-top:8px}
.tw-table{width:100%;border-collapse:collapse;font:11px/1.5 var(--tw-mono)}
.tw-table th,.tw-table td{padding:3px 5px;border-bottom:1px solid var(--tw-line);text-align:left;vertical-align:top}
.tw-table th{color:var(--tw-muted);font-weight:500}
.tw-trees{display:grid;gap:6px;margin-top:8px}
.tw-tree-root{display:grid;gap:3px;justify-items:start}
.tw-tree-root.selected{padding-left:6px;border-left:3px solid var(--tw-accent)}
.tw-tree{margin-left:18px;font:12px/1.6 var(--tw-mono)}
.tw-tree>summary{white-space:pre;cursor:pointer}
.tw-tree>.tw-tree,.tw-tree>.tw-tree-leaf{margin-left:18px}
.tw-tree-root>.tw-tree{margin-left:0}
.tw-tree-leaf{display:block;padding:0 0 0 14px;border:0;background:none;text-align:left;font:12px/1.6 var(--tw-mono);white-space:pre;color:var(--tw-ink);cursor:pointer}
.tw-detail{margin-top:14px;padding:12px;border:1px solid var(--tw-line);border-radius:4px;background:var(--tw-soft)}
.tw-detail h3{margin:0 0 8px;font-family:var(--tw-mono);font-size:13px}
.tw-detail-grid{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0 0 10px}
.tw-detail-grid dt{color:var(--tw-muted);font-size:11px}
.tw-detail-grid dd{margin:0;font:12px/1.5 var(--tw-mono);overflow-wrap:anywhere;white-space:pre-wrap}
.tw-row{display:flex;flex-wrap:wrap;gap:6px}
.tw-button{padding:3px 8px;border:1px solid var(--tw-line);border-radius:4px;background:var(--tw-card);color:var(--tw-ink);font-size:12px;cursor:pointer}
`;

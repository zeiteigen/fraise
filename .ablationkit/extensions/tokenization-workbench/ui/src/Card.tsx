/**
 * One configuration's result, as fraise-web's Tokenization view draws it (static/tokens.js):
 * segmentation strips, the composition icicle, the level table, composition trees, and a node
 * inspector. The data is fraise.preview.hierarchy.analyze_text's output, unchanged.
 */
import { useState } from "react";

export interface Token {
  index: number;
  id: number;
  text: string;
  start: number | null;
  end: number | null;
  base_start: number;
  base_end: number;
}
interface Cell {
  node: number;
  input_token: number;
  base_start: number;
  base_end: number;
  carried: boolean;
  introduced: number;
}
interface NodeInfo {
  node: number;
  token_id: number | null;
  text: string;
  atoms: number[];
  introduced: number;
  residual_width: number;
  residual_level: number;
  children: number[];
}
interface Tree {
  node: number;
  level: number;
  carried_through: number[];
  children: Tree[];
  text: string;
}
interface Level {
  level: number;
  name: string;
  vocab_size: number;
  new_nodes: number;
  max_composition_length?: number;
  mean_composition_length?: number;
  owned_nodes?: number;
  residual_dim?: number;
  residual_parameters?: number;
  residual_width_histogram?: Record<string, number>;
  distinct_nodes_in_text: number;
  input_vocabulary: boolean;
}
export interface Analysis {
  name: string;
  error?: string;
  byte_count: number;
  input_vocab_size: number;
  macro: boolean;
  base_vocab_size: number;
  base_tokens: { index: number; id: number; text: string }[];
  input_tokens: Token[];
  levels: Level[];
  rows: Cell[][];
  trees: Tree[];
  nodes: Record<string, NodeInfo>;
  residual_parameters?: number;
}

const COLORS = 6;
const num = (v: number | undefined, digits = 0) =>
  v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
export const visible = (text: string | null | undefined) =>
  String(text ?? "").replaceAll(" ", "·").replaceAll("\n", "↵").replaceAll("\t", "⇥") || "∅";

export function Card(props: { result: Analysis; index: number }) {
  const r = props.result;
  const [selected, setSelected] = useState<number | null>(null);
  const [focus, setFocus] = useState<number | null>(null);
  const [openTrees, setOpenTrees] = useState(false);

  const header = (
    <div className="tw-heading">
      <h2>{r.name || `Configuration ${props.index + 1}`}</h2>
      {r.levels?.length ? <span className="tw-badge">{r.levels.map((l) => l.name).join(" → ")}</span> : null}
    </div>
  );
  if (r.error)
    return (
      <article className="tw-card">
        {header}
        <p className="tw-error">{r.error}</p>
      </article>
    );

  const positions = r.input_tokens.length;
  const stats = [`${num(positions)} transformer positions (input tokens)`];
  if (r.macro)
    stats.push(
      `${num(r.base_tokens.length)} base tokens (output vocabulary)`,
      `${num(r.base_tokens.length / Math.max(positions, 1), 2)} base tokens/position`,
    );
  stats.push(`${num(r.byte_count / Math.max(positions, 1), 2)} bytes/position`);
  if (r.residual_parameters != null) stats.push(`${num(r.residual_parameters)} residual parameters in hierarchy`);

  const owner: number[] = [];
  for (const t of r.input_tokens) for (let i = t.base_start; i < t.base_end; i++) owner[i] = t.index;
  const hasComposition = r.rows?.length > 0;
  const selectInput = (index: number) => {
    setSelected(index);
    const top = r.rows?.[0]?.find((c) => c.input_token === index);
    if (top) setFocus(top.node);
    setOpenTrees(true);
  };
  const top = r.levels.length - 1;

  return (
    <article className="tw-card">
      {header}
      <p className="tw-metrics">{stats.join(" · ")}</p>
      {r.macro ? null : (
        <p className="tw-notice">
          Plain BPE tokenizer ({num(r.input_vocab_size)} tokens): input and output use the same vocabulary, so there
          is no base level and no hierarchy.
        </p>
      )}

      <section className="tw-section">
        <h3>Text segmentation · actual tokenization</h3>
        <p className="tw-help">
          Only the final input tokenizer creates transformer positions. Base tokens are what the local decoder emits.
        </p>
        <div className="tw-strip">
          <span className="tw-strip-label">Input tokens</span>
          {r.input_tokens.map((t) => (
            <button
              type="button"
              key={t.index}
              className={`tw-chip segment c${t.index % COLORS}${selected === t.index ? " selected" : ""}`}
              title={`input #${t.index} · id ${t.id} · ${t.base_end - t.base_start} base token(s)`}
              onClick={() => selectInput(t.index)}
            >
              {visible(t.text)}
            </button>
          ))}
        </div>
        {r.macro ? (
          <div className="tw-strip">
            <span className="tw-strip-label">Base tokens</span>
            {r.base_tokens.map((t) => (
              <span key={t.index} className={`tw-chip base c${(owner[t.index] ?? 0) % COLORS}`} title={`base #${t.index} · id ${t.id}`}>
                {visible(t.text)}
              </span>
            ))}
          </div>
        ) : null}
      </section>

      {hasComposition ? (
        <>
          <section className="tw-section">
            <h3>Representation composition · embedding only</h3>
            <p className="tw-help">
              Each row is a hierarchy level; a cell spans the base tokens it covers. Solid cells introduce a node;
              dashed cells carry a lower-level node by recomposing it from itself. Intermediate rows never become
              transformer positions.
            </p>
            <div className="tw-icicle-scroll">
              <div
                className="tw-icicle"
                style={{ gridTemplateColumns: `max-content repeat(${Math.max(r.base_tokens.length, 1)}, minmax(26px, max-content))` }}
              >
                {r.rows.map((row, rowIndex) => {
                  const level = top - rowIndex;
                  return [
                    <div key={`l${rowIndex}`} className="tw-level-label" style={{ gridRow: rowIndex + 1 }}>
                      {r.levels[level]?.name}
                    </div>,
                    ...row.map((cell) => {
                      const node = r.nodes[String(cell.node)];
                      return (
                        <button
                          type="button"
                          key={`${rowIndex}-${cell.base_start}`}
                          className={`tw-cell c${cell.input_token % COLORS}${cell.carried ? " carried" : ""}${
                            focus === cell.node ? " focused" : ""
                          }${selected === cell.input_token ? " selected" : ""}`}
                          style={{ gridRow: rowIndex + 1, gridColumn: `${cell.base_start + 2} / ${cell.base_end + 2}` }}
                          title={`node ${cell.node} · introduced at L${cell.introduced}${cell.carried ? " · carried" : ""} · residual ${node?.residual_width}`}
                          onClick={() => {
                            setSelected(cell.input_token);
                            setFocus(cell.node);
                          }}
                        >
                          <span className="tw-cell-text">{visible(node?.text)}</span>
                          <small>#{cell.node}</small>
                        </button>
                      );
                    }),
                  ];
                })}
              </div>
            </div>
          </section>

          <details className="tw-section" open>
            <summary>Levels, residual ownership, and parameters</summary>
            <div className="tw-table-scroll">
            <table className="tw-table">
              <thead>
                <tr>
                  {["Level", "Vocab", "New nodes", "Children (max · mean)", "Owned residuals", "Max width", "Residual params", "Width histogram", "Distinct in text"].map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {r.levels.map((l) => (
                  <tr key={l.level}>
                    <td>{`${l.name}${l.input_vocabulary ? " · input" : l.level === 0 ? " · atoms" : ""}`}</td>
                    <td>{num(l.vocab_size)}</td>
                    <td>{num(l.new_nodes)}</td>
                    <td>{l.level ? `${l.max_composition_length} · ${num(l.mean_composition_length, 2)}` : "—"}</td>
                    <td>{l.level ? num(l.owned_nodes) : "— (owned by L1)"}</td>
                    <td>{l.level ? l.residual_dim : "—"}</td>
                    <td>{l.level ? num(l.residual_parameters) : "—"}</td>
                    <td>
                      {l.residual_width_histogram
                        ? Object.entries(l.residual_width_histogram).map(([w, c]) => `${w}:${num(c)}`).join(" ")
                        : "—"}
                    </td>
                    <td>{num(l.distinct_nodes_in_text)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
            <p className="tw-help">
              Every node owns one residual, added at the level that introduces it (atoms at L1). Nodes carried to
              higher levels reuse it; no level duplicates residual parameters.
            </p>
          </details>

          <details className="tw-section" open={openTrees} onToggle={(e) => setOpenTrees((e.target as HTMLDetailsElement).open)}>
            <summary>Composition trees</summary>
            <div className="tw-trees">
              {r.trees.map((tree, index) => {
                const token = r.input_tokens[index];
                return (
                  <div key={index} className={`tw-tree-root${selected === index ? " selected" : ""}`}>
                    <span className={`tw-chip segment c${index % COLORS}`}>
                      #{index} · {visible(token?.text)} · id {token?.id}
                    </span>
                    <TreeNode tree={tree} nodes={r.nodes} open={selected === index} onPick={setFocus} />
                  </div>
                );
              })}
            </div>
          </details>

          <NodeDetail result={r} id={focus} token={selected == null ? null : (r.input_tokens[selected] ?? null)} onPick={setFocus} />
        </>
      ) : null}
    </article>
  );
}

function TreeNode(props: { tree: Tree; nodes: Record<string, NodeInfo>; open: boolean; onPick: (n: number) => void }) {
  const { tree, nodes } = props;
  const node = nodes[String(tree.node)];
  const label = `${visible(tree.text)}  node ${tree.node} · L${tree.level}${
    tree.carried_through.length ? ` (carried through L${tree.carried_through.join(", L")})` : ""
  } · residual ${node?.residual_width}@L${node?.residual_level}`;
  if (!tree.children.length)
    return (
      <button type="button" className="tw-tree-leaf" onClick={() => props.onPick(tree.node)}>
        {label}
      </button>
    );
  return (
    <details className="tw-tree" open={props.open}>
      <summary onClick={() => props.onPick(tree.node)}>{label}</summary>
      {tree.children.map((child, i) => (
        <TreeNode key={i} tree={child} nodes={nodes} open={props.open} onPick={props.onPick} />
      ))}
    </details>
  );
}

function NodeDetail(props: { result: Analysis; id: number | null; token: Token | null; onPick: (n: number) => void }) {
  const { result, id, token } = props;
  const node = id == null ? undefined : result.nodes[String(id)];
  if (!node)
    return (
      <div className="tw-detail">
        <p className="tw-help">Select a token or node to inspect its ID, level, residual, and composition.</p>
      </div>
    );
  const rows: [string, string][] = [];
  if (token) rows.push(["Input position", `#${token.index} · characters ${token.start ?? "—"}–${token.end ?? "—"}`]);
  rows.push(
    ["Input token ID", node.token_id == null ? "none (intermediate representation node)" : String(node.token_id)],
    ["Introduced at", `${result.levels[node.introduced]?.name}${node.introduced === 0 ? " (atom)" : ""}`],
    ["Present at", result.levels.filter((l) => l.level >= node.introduced).map((l) => l.name).join(", ")],
    ["Residual", node.residual_width ? `width ${node.residual_width}, added once at L${node.residual_level}` : "none (width 0)"],
    [
      "Composed from",
      node.children.length
        ? node.children.map((c) => `${c} ${visible(result.nodes[String(c)]?.text ?? "")}`).join("  +  ")
        : node.introduced === 0
          ? "atom embedding"
          : "—",
    ],
    ["Atoms (base IDs)", node.atoms.join(" ")],
  );
  return (
    <div className="tw-detail">
      <h3>
        Node {id} · {visible(node.text)}
      </h3>
      <dl className="tw-detail-grid">
        {rows.map(([term, value]) => [<dt key={`t${term}`}>{term}</dt>, <dd key={`d${term}`}>{value}</dd>])}
      </dl>
      {node.children.length ? (
        <div className="tw-row">
          {node.children.map((c) => (
            <button type="button" key={c} className="tw-button" onClick={() => props.onPick(c)}>
              Inspect {c}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

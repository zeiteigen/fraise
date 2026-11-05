import { post, request } from '/static/api.js';
import { $, $$, element, option, parseRef, populateCheckpoints, formatNumber } from '/static/ui.js';

const KINDS = [
  ['auto', 'Tokenizer default (two-level)'],
  ['merges', 'Merge-prefix levels'],
  ['file', 'Hierarchy file'],
  ['run', 'Saved run (trained hierarchy)'],
  ['none', 'Segmentation only'],
];
const COLORS = 6;

function visible(text) {
  return String(text ?? '').replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥') || '∅';
}

function numbers(value) {
  const parts = String(value || '').split(/[\s,→>-]+/).filter(Boolean).map(Number);
  return parts.length && parts.every(Number.isInteger) ? parts : null;
}

export class Tokens {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.hierarchies = [];
    this.results = [];
    this.selection = null;
    root.innerHTML = `<div class="page-heading"><div><h1>Tokenization and hierarchy</h1><p>Segmentation decides transformer positions; composition only builds each input token's embedding from smaller tokens.</p></div></div>
      <form id="tok-form" class="panel"><div class="panel-heading"><h2>Text</h2><label class="check-label"><input type="checkbox" id="tok-special"> Include BOS/EOS</label></div>
        <label class="sr-only" for="tok-text">Text to tokenize</label><textarea id="tok-text" class="compare-prompt" placeholder="Type any text…">The strawberry fields were unbelievably photosynthesizing.</textarea>
        <div class="panel-heading tok-config-heading"><h2>Hierarchy configurations</h2><button type="button" class="text-button" id="tok-add">Add configuration to compare</button></div>
        <div id="tok-configs" class="tok-configs"></div>
        <div class="compare-form-footer"><p>Updates as you type. Level sizes start at the base vocabulary, e.g. <code>8192 16384 65536</code>; residual widths are one per composed level.</p><button type="submit" class="primary-button">Analyze</button></div>
        <p class="inline-error" id="tok-error" role="alert" hidden></p></form>
      <div id="tok-results" class="tok-results" aria-live="polite"></div>`;
    this.form = $('#tok-form', root);
    this.form.addEventListener('submit', event => { event.preventDefault(); this.analyze(); });
    this.form.addEventListener('input', () => this.schedule());
    this.form.addEventListener('change', () => this.schedule());
    $('#tok-add', root).addEventListener('click', () => { this.addConfiguration(); this.schedule(); });
    this.addConfiguration();
  }

  updateBusy() {}

  async refresh() {
    try { this.hierarchies = (await request('/api/hierarchies')).hierarchies || []; } catch { this.hierarchies = []; }
    $$('.tok-config', this.root).forEach(row => this.populate(row));
    if (!this.results.length) this.schedule();
  }

  addConfiguration() {
    if ($$('.tok-config', this.root).length >= 6) { this.app.notify('Compare up to six configurations.'); return; }
    const row = element('div', 'tok-config');
    row.innerHTML = `<label>Hierarchy<select name="kind">${KINDS.map(([value, label]) => `<option value="${value}">${label}</option>`).join('')}</select></label>
      <label data-for="tokenizer">Tokenizer<select name="tokenizer"></select></label>
      <label data-for="merges">Level sizes<input name="levels" placeholder="8192 16384 65536"></label>
      <label data-for="merges auto">Residual widths<input name="residuals" placeholder="32" value="32"></label>
      <label data-for="file">Hierarchy file<select name="file"></select></label>
      <label data-for="run">Run checkpoint<select name="run"></select></label>
      <button type="button" class="text-button cancel-button" name="remove">Remove</button>`;
    $('[name="kind"]', row).addEventListener('change', () => this.toggleFields(row));
    $('[name="remove"]', row).addEventListener('click', () => {
      if ($$('.tok-config', this.root).length > 1) { row.remove(); this.schedule(); }
    });
    $('#tok-configs', this.root).append(row);
    this.populate(row);
    this.toggleFields(row);
  }

  populate(row) {
    const catalog = this.app.catalog;
    const tokenizer = $('[name="tokenizer"]', row);
    const current = tokenizer.value;
    const describe = item => item.macro
      ? `${item.label} · macro ${formatNumber(item.vocab_size, 0)} over base ${formatNumber(item.base_vocab_size, 0)}`
      : `${item.label} · plain BPE ${formatNumber(item.vocab_size, 0)} (no hierarchy)`;
    tokenizer.replaceChildren(...(catalog?.tokenizers || []).map(item => option(item.id, describe(item))));
    if ([...tokenizer.options].some(item => item.value === current)) tokenizer.value = current;
    else {
      // Prefer a macro tokenizer: only those have a hierarchy to show.
      const ranked = [...(catalog?.tokenizers || [])].sort((a, b) => Number(b.macro) - Number(a.macro) || b.vocab_size - a.vocab_size);
      if (ranked[0]) tokenizer.value = ranked[0].id;
    }
    const file = $('[name="file"]', row);
    const selectedFile = file.value;
    file.replaceChildren(option('', this.hierarchies.length ? 'Select a file' : 'No files in DATA_DIR/hierarchies'), ...this.hierarchies.map(item => option(item.id, item.label)));
    file.value = selectedFile;
    populateCheckpoints($('[name="run"]', row), this.app.runs);
  }

  toggleFields(row) {
    const kind = $('[name="kind"]', row).value;
    $$('[data-for]', row).forEach(label => { label.hidden = !label.dataset.for.split(' ').includes(kind); });
    $('[data-for="tokenizer"]', row).hidden = kind === 'run';
  }

  configurations() {
    return $$('.tok-config', this.root).map(row => {
      const kind = $('[name="kind"]', row).value;
      const hierarchy = { kind };
      if (kind === 'merges') hierarchy.level_sizes = numbers($('[name="levels"]', row).value);
      if (kind === 'merges' || kind === 'auto') hierarchy.residual_dims = numbers($('[name="residuals"]', row).value);
      if (kind === 'file') hierarchy.file_id = $('[name="file"]', row).value || null;
      if (kind === 'run') hierarchy.run = parseRef($('[name="run"]', row).value);
      return { tokenizer_id: kind === 'run' ? null : $('[name="tokenizer"]', row).value || null, hierarchy };
    });
  }

  schedule() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.analyze(), 350);
  }

  async analyze() {
    const text = $('#tok-text', this.root).value;
    const error = $('#tok-error', this.root);
    error.hidden = true;
    if (!text || !this.app.catalog) return;
    const sequence = (this.sequence || 0) + 1;
    this.sequence = sequence;
    try {
      const data = await post('/api/tokenize', { text, add_special_tokens: $('#tok-special', this.root).checked, configurations: this.configurations() });
      if (sequence !== this.sequence) return;
      this.results = data.results;
      this.render();
    } catch (failure) {
      if (sequence !== this.sequence) return;
      error.textContent = failure.message;
      error.hidden = false;
    }
  }

  render() {
    const container = $('#tok-results', this.root);
    container.replaceChildren(...this.results.map((result, index) => this.card(result, index)));
  }

  card(result, index) {
    const card = element('article', 'panel tok-card');
    const heading = element('div', 'panel-heading');
    heading.append(element('h2', '', result.name || `Configuration ${index + 1}`));
    if (result.levels?.length) heading.append(element('span', 'badge', result.levels.map(level => level.name).join(' → ')));
    card.append(heading);
    if (result.error) { card.append(element('p', 'inline-error', result.error)); return card; }
    const positions = result.input_tokens.length;
    const stats = [
      `${formatNumber(positions, 0)} transformer positions (input tokens)`,
      `${formatNumber(result.byte_count / Math.max(positions, 1), 2)} bytes/position`,
    ];
    if (result.macro) stats.splice(1, 0, `${formatNumber(result.base_tokens.length, 0)} base tokens (output vocabulary)`, `${formatNumber(result.base_tokens.length / Math.max(positions, 1), 2)} base tokens/position`);
    if (result.residual_parameters != null) stats.push(`${formatNumber(result.residual_parameters, 0)} residual parameters in hierarchy`);
    card.append(element('p', 'metrics', stats.join(' · ')));
    if (!result.macro) card.append(element('p', 'form-notice', `This is a plain BPE tokenizer (${formatNumber(result.input_vocab_size, 0)} tokens): input and output use the same vocabulary, so there is no base level and no hierarchy. Select a macro tokenizer (created with create-tokenizer --base-tokenizer) to see composition.`));
    const state = { result, selected: null, card };
    card.append(this.segmentation(state));
    if (result.rows?.length) {
      card.append(this.composition(state), this.levelTable(result), this.trees(state));
      state.detail = element('div', 'tok-detail');
      state.detail.append(element('p', 'subtle', 'Select a token or node to inspect its ID, level, residual, and composition.'));
      card.append(state.detail);
    }
    return card;
  }

  segmentation(state) {
    const { result } = state;
    const section = element('section', 'tok-section');
    section.append(element('h3', '', 'Text segmentation · actual tokenization'), element('p', 'field-help', 'Only the final input tokenizer creates transformer positions. Base tokens are what the local decoder emits.'));
    const inputs = element('div', 'tok-strip');
    inputs.append(element('span', 'tok-strip-label', 'Input tokens'));
    for (const token of result.input_tokens) {
      const chip = element('button', `tok-chip segment color-${token.index % COLORS}`, visible(token.text));
      chip.type = 'button';
      chip.title = `input #${token.index} · id ${token.id} · ${token.base_end - token.base_start} base token(s)`;
      chip.dataset.input = token.index;
      chip.addEventListener('click', () => this.selectInput(state, token.index));
      inputs.append(chip);
    }
    const bases = element('div', 'tok-strip');
    bases.append(element('span', 'tok-strip-label', 'Base tokens'));
    const owner = [];
    for (const token of result.input_tokens) for (let i = token.base_start; i < token.base_end; i++) owner[i] = token.index;
    for (const token of result.base_tokens) {
      const chip = element('span', `tok-chip base color-${(owner[token.index] ?? 0) % COLORS}`, visible(token.text));
      chip.title = `base #${token.index} · id ${token.id}`;
      bases.append(chip);
    }
    section.append(inputs);
    if (result.macro) section.append(bases);
    return section;
  }

  composition(state) {
    const { result } = state;
    const section = element('section', 'tok-section');
    section.append(element('h3', '', 'Representation composition · embedding only'), element('p', 'field-help', 'Each row is a hierarchy level; a cell spans the base tokens it covers. Solid cells introduce a node; dashed cells carry a lower-level node by recomposing it from itself. Intermediate rows never become transformer positions.'));
    const scroller = element('div', 'tok-icicle-scroll');
    const grid = element('div', 'tok-icicle');
    grid.style.gridTemplateColumns = `max-content repeat(${Math.max(result.base_tokens.length, 1)}, minmax(26px, max-content))`;
    const top = result.levels.length - 1;
    result.rows.forEach((row, rowIndex) => {
      const level = top - rowIndex;
      const label = element('div', 'tok-level-label', result.levels[level].name);
      label.style.gridRow = rowIndex + 1;
      grid.append(label);
      for (const cell of row) {
        const node = result.nodes[cell.node];
        const button = element('button', `tok-cell color-${cell.input_token % COLORS}${cell.carried ? ' carried' : ''}`);
        button.type = 'button';
        button.style.gridRow = rowIndex + 1;
        button.style.gridColumn = `${cell.base_start + 2} / ${cell.base_end + 2}`;
        button.dataset.input = cell.input_token;
        button.dataset.node = cell.node;
        button.append(element('span', 'tok-cell-text', visible(node.text)), element('small', '', `#${cell.node}`));
        button.title = `node ${cell.node} · introduced at L${cell.introduced}${cell.carried ? ' · carried' : ''} · residual ${node.residual_width}`;
        button.addEventListener('click', () => { this.selectInput(state, cell.input_token); this.showNode(state, cell.node); });
        grid.append(button);
      }
    });
    scroller.append(grid);
    section.append(scroller);
    return section;
  }

  levelTable(result) {
    const section = element('details', 'tok-section tok-levels');
    section.open = true;
    section.append(element('summary', '', 'Levels, residual ownership, and parameters'));
    const table = element('table', 'tok-table');
    table.innerHTML = '<thead><tr><th>Level</th><th>Vocab</th><th>New nodes</th><th>Children (max · mean)</th><th>Owned residuals</th><th>Max width</th><th>Residual params</th><th>Width histogram</th><th>Distinct in text</th></tr></thead>';
    const body = element('tbody');
    for (const level of result.levels) {
      const row = element('tr');
      const histogram = level.residual_width_histogram ? Object.entries(level.residual_width_histogram).map(([width, count]) => `${width}:${formatNumber(count, 0)}`).join(' ') : '—';
      const cells = [
        `${level.name}${level.input_vocabulary ? ' · input' : level.level === 0 ? ' · atoms' : ''}`,
        formatNumber(level.vocab_size, 0), formatNumber(level.new_nodes, 0),
        level.level ? `${level.max_composition_length} · ${formatNumber(level.mean_composition_length, 2)}` : '—',
        level.level ? formatNumber(level.owned_nodes, 0) : '— (owned by L1)',
        level.level ? level.residual_dim : '—',
        level.level ? formatNumber(level.residual_parameters, 0) : '—',
        histogram, formatNumber(level.distinct_nodes_in_text, 0),
      ];
      for (const value of cells) row.append(element('td', '', String(value)));
      body.append(row);
    }
    table.append(body);
    section.append(table, element('p', 'field-help', 'Every node owns one residual, added at the level that introduces it (atoms at L1). Nodes carried to higher levels reuse it; no level duplicates residual parameters.'));
    return section;
  }

  trees(state) {
    const { result } = state;
    const section = element('details', 'tok-section');
    section.append(element('summary', '', 'Composition trees'));
    const toolbar = element('div', 'button-row');
    const expand = element('button', 'text-button', 'Expand all');
    expand.type = 'button';
    const collapse = element('button', 'text-button', 'Collapse all');
    collapse.type = 'button';
    toolbar.append(expand, collapse);
    const list = element('div', 'tok-trees');
    expand.addEventListener('click', () => $$('details', list).forEach(item => { item.open = true; }));
    collapse.addEventListener('click', () => $$('details', list).forEach(item => { item.open = false; }));
    result.trees.forEach((tree, index) => {
      const token = result.input_tokens[index];
      const wrapper = element('div', 'tok-tree-root');
      wrapper.dataset.input = index;
      wrapper.append(element('span', `tok-chip segment color-${index % COLORS}`, `#${index} · ${visible(token.text)} · id ${token.id}`), this.treeNode(state, tree));
      list.append(wrapper);
    });
    section.append(toolbar, list);
    state.treeSection = section;
    return section;
  }

  treeNode(state, tree) {
    const node = state.result.nodes[tree.node];
    const label = `${visible(tree.text)}  node ${tree.node} · L${tree.level}${tree.carried_through.length ? ` (carried through L${tree.carried_through.join(', L')})` : ''} · residual ${node.residual_width}@L${node.residual_level}`;
    if (!tree.children.length) {
      const leaf = element('button', 'tok-tree-leaf', label);
      leaf.type = 'button';
      leaf.addEventListener('click', () => this.showNode(state, tree.node));
      return leaf;
    }
    const details = element('details', 'tok-tree');
    const summary = element('summary', '', label);
    summary.addEventListener('click', () => this.showNode(state, tree.node));
    details.append(summary, ...tree.children.map(child => this.treeNode(state, child)));
    return details;
  }

  selectInput(state, index) {
    state.selected = index;
    $$('[data-input]', state.card).forEach(item => item.classList.toggle('selected', Number(item.dataset.input) === index));
    const token = state.result.input_tokens[index];
    const top = state.result.rows?.[0]?.find(cell => cell.input_token === index);
    if (top) this.showNode(state, top.node, token);
    const root = $(`.tok-tree-root[data-input="${index}"]`, state.card);
    if (root && state.treeSection) { state.treeSection.open = true; $$('details', root).forEach(item => { item.open = true; }); }
  }

  showNode(state, id, token = null) {
    if (!state.detail) return;
    const { result } = state;
    const node = result.nodes[id];
    const detail = state.detail;
    detail.replaceChildren();
    detail.append(element('h3', '', `Node ${id} · ${visible(node.text)}`));
    const list = element('dl', 'tok-detail-grid');
    const add = (term, value) => list.append(element('dt', '', term), element('dd', '', value));
    if (token) add('Input position', `#${token.index} · characters ${token.start ?? '—'}–${token.end ?? '—'}`);
    add('Input token ID', node.token_id == null ? 'none (intermediate representation node)' : String(node.token_id));
    add('Introduced at', `${result.levels[node.introduced].name}${node.introduced === 0 ? ' (atom)' : ''}`);
    add('Present at', result.levels.filter(level => level.level >= node.introduced).map(level => level.name).join(', '));
    add('Residual', node.residual_width ? `width ${node.residual_width}, added once at L${node.residual_level}` : 'none (width 0)');
    add('Composed from', node.children.length ? node.children.map(child => `${child} ${visible(result.nodes[child]?.text ?? '')}`).join('  +  ') : node.introduced === 0 ? 'atom embedding' : '—');
    add('Atoms (base IDs)', node.atoms.join(' '));
    detail.append(list);
    if (node.children.length) {
      const row = element('div', 'button-row');
      for (const child of node.children) {
        const button = element('button', 'secondary-button', `Inspect ${child}`);
        button.type = 'button';
        button.addEventListener('click', () => this.showNode(state, child));
        row.append(button);
      }
      detail.append(row);
    }
    $$('.tok-cell', state.card).forEach(cell => cell.classList.toggle('focused', cell.dataset.node === String(id)));
  }
}

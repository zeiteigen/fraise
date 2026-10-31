export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function option(value, label) {
  const item = document.createElement('option');
  item.value = value;
  item.textContent = label;
  return item;
}

export function refKey(ref) { return JSON.stringify([ref.run_id, ref.checkpoint_id]); }
export function parseRef(value) {
  if (!value) return null;
  try { const [run_id, checkpoint_id] = JSON.parse(value); return { run_id, checkpoint_id }; } catch { return null; }
}

export function checkpointLabel(run, checkpoint) {
  return `${run.name || run.id} · ${checkpoint.label || checkpoint.id}`;
}

export function populateCheckpoints(select, runs, value = select.value, placeholder = 'Select a checkpoint') {
  select.replaceChildren(option('', placeholder));
  for (const run of runs) {
    if (!run.checkpoints?.length) continue;
    const group = document.createElement('optgroup');
    group.label = run.name || run.id;
    for (const checkpoint of run.checkpoints) {
      group.append(option(refKey({ run_id: run.id, checkpoint_id: checkpoint.id }), checkpoint.label || checkpoint.id));
    }
    select.append(group);
  }
  if ([...select.options].some(item => item.value === value)) select.value = value;
}

export function formatNumber(value, digits = 1) {
  return Number.isFinite(Number(value)) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(Number(value)) : '—';
}

export function settingsMarkup(prefix) {
  return `<details class="sampling"><summary>Generation settings <span class="summary-hint">128 tokens · greedy</span></summary><div class="settings-grid">
    <label>Max new tokens<input id="${prefix}-tokens" name="max_new_tokens" type="number" min="1" max="4096" step="1" value="128" required></label>
    <label>Temperature<input id="${prefix}-temperature" name="temperature" type="number" min="0" max="100" step="0.05" value="0" required></label>
    <label>Top p<input id="${prefix}-top-p" name="top_p" type="number" min="0.000001" max="1" step="any" value="0.95" placeholder="Optional"></label>
    <label>Top k<input id="${prefix}-top-k" name="top_k" type="number" min="1" step="1" placeholder="Optional"></label>
    <label>Seed<input id="${prefix}-seed" name="seed" type="number" min="0" step="1" value="1337" required></label>
  </div><p class="field-help">Temperature 0 uses greedy decoding. Token counts depend on each model’s tokenizer.</p></details>`;
}

export function readSettings(root) {
  const number = name => {
    const input = root.querySelector(`[name="${name}"]`);
    return input.value === '' ? null : Number(input.value);
  };
  return { max_new_tokens: number('max_new_tokens'), temperature: number('temperature'), top_p: number('top_p'), top_k: number('top_k'), seed: number('seed') };
}

export function wireSettings(root) {
  const details = $('.sampling', root);
  details.addEventListener('input', () => {
    const values = readSettings(root);
    $('.summary-hint', details).textContent = `${values.max_new_tokens ?? '—'} tokens · ${values.temperature === 0 ? 'greedy' : `temp ${values.temperature}`}`;
  });
}

export function restoreSettings(root, values = {}) {
  for (const [name, value] of Object.entries(values)) {
    const input = root.querySelector(`[name="${name}"]`);
    if (input) input.value = value ?? '';
  }
  root.querySelector('.sampling')?.dispatchEvent(new Event('input'));
}

export function enterToSubmit(textarea, form) {
  textarea.addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
      event.preventDefault();
      if (!form.querySelector('[type="submit"]').disabled) form.requestSubmit();
    }
  });
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  });
}

export async function copyText(text, notify) {
  try { await navigator.clipboard.writeText(text); notify('Copied to clipboard'); }
  catch { notify('Clipboard unavailable. Select the text to copy it.'); }
}

export function metricsText(result) {
  const parts = [];
  if (result.generated_tokens != null) parts.push(`${formatNumber(result.generated_tokens, 0)} tokens`);
  if (result.seconds != null) parts.push(`${formatNumber(result.seconds, 2)}s`);
  if (result.tokens_per_second != null) parts.push(`${formatNumber(result.tokens_per_second)} tok/s`);
  if (result.load_seconds != null) parts.push(`${formatNumber(result.load_seconds, 2)}s load`);
  if (result.prompt_tokens != null) parts.push(`${formatNumber(result.prompt_tokens, 0)} prompt tokens`);
  if (result.finish_reason) parts.push(String(result.finish_reason).replaceAll('_', ' '));
  return parts.join(' · ');
}

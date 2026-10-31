import { generate } from '/static/api.js';
import { $, $$, element, refKey, parseRef, settingsMarkup, wireSettings, readSettings, enterToSubmit, copyText, metricsText, formatNumber } from '/static/ui.js';

export class Compare {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.controller = null;
    this.cards = [];
    root.innerHTML = `<div class="page-heading"><div><h1>Compare checkpoints</h1><p>Run the same text and sampling settings across selected checkpoints.</p></div></div>
      <form id="compare-form"><div class="panel"><div class="panel-heading"><h2>Checkpoints</h2><span class="badge" id="compare-count">0 selected</span></div><div class="select-list" id="compare-models"></div></div>
      <div class="panel"><div class="panel-heading"><h2>Input text</h2></div><label class="sr-only" for="compare-prompt">Shared input text</label><textarea class="compare-prompt" id="compare-prompt" placeholder="Text to continue…" required></textarea>${settingsMarkup('compare')}<div class="compare-form-footer"><p>Checkpoints run sequentially. Ctrl/Cmd+Enter to run; Enter for a newline.</p><div class="button-row"><button type="button" id="compare-cancel" class="text-button cancel-button" hidden>Stop comparison</button><button type="submit" class="primary-button">Compare checkpoints</button></div></div><p class="inline-error" id="compare-error" role="alert" hidden></p></div></form>
      <div class="compare-results" id="compare-results" aria-label="Comparison results"></div><p class="compare-note" id="compare-note" hidden>Each checkpoint uses its saved configuration and tokenizer. Token counts, speed, and validation losses can differ with the tokenizer and are not always directly comparable.</p>`;
    this.form = $('#compare-form', root);
    this.form.addEventListener('submit', event => { event.preventDefault(); this.send(); });
    $('#compare-cancel', root).addEventListener('click', () => this.controller?.abort());
    wireSettings(root);
    enterToSubmit($('#compare-prompt', root), this.form);
  }

  selectedRefs() { return $$('input[name="checkpoint"]:checked', this.root).map(input => parseRef(input.value)).filter(Boolean); }

  refresh() {
    const selected = new Set(this.selectedRefs().map(refKey));
    const list = $('#compare-models', this.root);
    list.replaceChildren();
    for (const run of this.app.runs) {
      if (!run.checkpoints?.length) continue;
      const group = element('div', 'run-select-group');
      const heading = element('div', 'run-select-heading');
      const name = element('div');
      const parameters = run.parameter_count == null ? '—' : formatNumber(run.parameter_count, 0);
      const context = run.config?.max_seq_len == null ? '—' : formatNumber(run.config.max_seq_len, 0);
      const vocabulary = run.tokenizer?.vocab_size == null ? '—' : formatNumber(run.tokenizer.vocab_size, 0);
      name.append(element('strong', '', run.name || run.id), element('small', 'run-metadata', `${run.architecture || 'Unknown architecture'} · ${parameters} parameters · context ${context} · vocab ${vocabulary} · saved step ${formatNumber(run.global_step ?? 0, 0)}`));
      heading.append(name, element('span', 'badge', run.finished ? 'Finished' : 'Saved run'));
      const choices = element('div', 'checkpoint-options');
      for (const checkpoint of run.checkpoints) {
        const label = element('label', 'checkpoint-chip');
        const input = document.createElement('input');
        input.type = 'checkbox'; input.name = 'checkpoint';
        input.value = refKey({ run_id: run.id, checkpoint_id: checkpoint.id });
        input.checked = selected.has(input.value);
        input.addEventListener('change', () => {
          if (this.selectedRefs().length > 8) { input.checked = false; this.app.notify('Choose up to eight checkpoints per comparison.'); }
          this.updateBusy();
        });
        label.append(input, element('span', '', checkpoint.label || checkpoint.id));
        choices.append(label);
      }
      group.append(heading, choices, this.runDetails(run));
      list.append(group);
    }
    if (!list.children.length) {
      const empty = element('div', 'empty-state');
      empty.append(element('h3', '', 'No checkpoints available'), element('p', '', this.app.catalog ? 'Train a model or add a saved run to compare checkpoints.' : 'Loading saved checkpoints…'));
      const train = element('button', 'text-button accent-text', 'Open training');
      train.type = 'button'; train.addEventListener('click', () => this.app.navigate('train')); empty.append(train); list.append(empty);
    }
    this.updateBusy();
  }

  runDetails(run) {
    const details = element('details', 'run-details');
    details.append(element('summary', '', 'Saved model, tokenizer & validation'));
    const identity = element('dl');
    for (const [title, value] of [['Tokenizer', run.tokenizer?.id || 'Archived tokenizer'], ['Vocabulary', run.tokenizer?.vocab_size ? `${formatNumber(run.tokenizer.vocab_size, 0)} tokens` : 'Unavailable'], ['Fingerprint', run.tokenizer?.fingerprint || 'Unavailable']]) {
      const pair = element('div'); pair.append(element('dt', '', title), element('dd', '', value)); identity.append(pair);
    }
    details.append(identity, element('p', '', 'Validation loss and token counts depend on the saved tokenizer.'));
    const data = { model_config: run.config, training_config: run.training, validation: run.validation, validation_results: run.validation_results };
    details.append(element('pre', '', JSON.stringify(data, null, 2)));
    return details;
  }

  updateBusy() {
    const count = this.selectedRefs().length;
    $('#compare-count', this.root).textContent = `${count} selected`;
    $('button[type="submit"]', this.root).disabled = Boolean(this.app.busy) || count === 0;
    $('#compare-cancel', this.root).hidden = !this.controller;
    $$('input, textarea', this.root).forEach(input => { input.disabled = Boolean(this.controller); });
  }

  createCard(ref) {
    const run = this.app.runs.find(run => run.id === ref.run_id);
    const checkpoint = run?.checkpoints.find(checkpoint => checkpoint.id === ref.checkpoint_id);
    const article = element('article', 'result-card');
    const header = element('div', 'result-card-header');
    const title = element('div');
    title.append(element('h3', '', run?.name || ref.run_id), element('small', '', checkpoint?.label || ref.checkpoint_id));
    const status = element('span', 'badge', 'Queued');
    header.append(title, status);
    const output = element('div', 'result-card-body');
    const footer = element('div', 'result-card-footer');
    const metrics = element('span', 'metrics', 'Waiting for earlier checkpoints');
    const copy = element('button', 'copy-button', 'Copy');
    copy.setAttribute('aria-label', `Copy output from ${run?.name || ref.run_id}`);
    copy.addEventListener('click', () => copyText(output.textContent, text => this.app.notify(text)));
    footer.append(metrics, copy); article.append(header, output, footer);
    $('#compare-results', this.root).append(article);
    return { article, output, metrics, status, done: false };
  }

  async send() {
    const models = this.selectedRefs();
    const prompt = $('#compare-prompt', this.root).value;
    if (!models.length || !prompt.trim() || this.app.busy || !this.form.reportValidity()) return;
    const errorNode = $('#compare-error', this.root);
    errorNode.hidden = true;
    $('#compare-results', this.root).replaceChildren();
    $('#compare-note', this.root).hidden = false;
    this.cards = models.map(ref => this.createCard(ref));
    this.controller = new AbortController();
    this.app.setLocalOperation({ kind: 'generation', id: 'compare' });
    try {
      await generate({ models, prompt, ...readSettings(this.root) }, (event, data) => {
        const card = this.cards[data.index];
        if (event === 'error') {
          if (!card) throw new Error(data.message || 'Comparison failed.');
          card.status.textContent = 'Failed'; card.status.className = 'badge failed';
          card.metrics.textContent = data.message || 'Generation failed'; card.done = true;
          card.article.classList.remove('streaming');
        }
        if (!card) return;
        if (event === 'loading') {
          card.article.classList.add('streaming'); card.status.textContent = 'Loading'; card.status.className = 'badge running'; card.metrics.textContent = 'Loading checkpoint…';
        } else if (event === 'output' || event === 'result') {
          card.output.textContent = data.text || '';
          card.status.textContent = event === 'result' ? 'Complete' : 'Generating';
          card.metrics.textContent = event === 'result' ? metricsText(data) : `${data.generated_tokens || 0} tokens · generating`;
          if (event === 'result') { card.done = true; card.status.className = 'badge success'; card.article.classList.remove('streaming'); }
        }
      }, this.controller.signal);
    } catch (error) {
      const stopped = error.name === 'AbortError';
      if (!stopped) { errorNode.textContent = error.message; errorNode.hidden = false; }
      for (const card of this.cards.filter(card => !card.done)) {
        card.status.textContent = stopped ? 'Stopped' : 'Interrupted';
        card.status.className = 'badge';
        card.metrics.textContent = card.output.textContent ? 'Partial output kept' : stopped ? 'Comparison stopped' : 'Could not finish generation';
        card.article.classList.remove('streaming');
      }
    } finally {
      this.controller = null;
      this.app.setLocalOperation(null);
      await this.app.pollStatus();
    }
  }
}

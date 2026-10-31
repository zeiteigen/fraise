import { generate } from '/static/api.js';
import { $, parseRef, refKey, populateCheckpoints, settingsMarkup, wireSettings, readSettings, restoreSettings, copyText, metricsText, formatNumber } from '/static/ui.js';
import { SavedInputs } from '/static/saved-inputs.js';
import { renderRunInformation } from '/static/run-information.js';

export class Autocomplete {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.controller = null;
    this.id = null;
    this.suggestion = '';
    this.metrics = null;
    this.replacement = null;
    root.innerHTML = `<div class="autocomplete-heading page-heading"><h1>Autocomplete</h1><p>Generate a continuation from the text before the cursor. Accept it to insert it into the document.</p></div>
      <div class="autocomplete-grid"><div class="autocomplete-main"><form id="autocomplete-form" class="panel">
        <div class="panel-heading"><h2>Input text</h2><label class="checkpoint-control" for="autocomplete-checkpoint">Checkpoint <select id="autocomplete-checkpoint"><option value="">Select a checkpoint</option></select></label></div>
        <label class="sr-only" for="autocomplete-text">Text to complete</label><textarea id="autocomplete-text" class="document-editor" rows="14" maxlength="100000" spellcheck="false" placeholder="Enter text, place the cursor, then generate a continuation."></textarea>
        <div class="editor-status"><span id="autocomplete-position">0 characters</span><span>Ctrl/Cmd+Enter: generate · Tab: accept · Esc: discard</span></div>
        ${settingsMarkup('autocomplete')}
        <div class="editor-toolbar"><span id="autocomplete-status" role="status">Select a checkpoint and enter text.</span><div class="button-row"><button type="button" id="autocomplete-cancel" class="text-button" hidden>Stop generation</button><button type="submit" class="primary-button" id="autocomplete-generate">Generate continuation</button></div></div>
        <p id="autocomplete-error" class="inline-error" role="alert" hidden></p>
      </form><section class="panel" aria-label="Suggested continuation"><div class="panel-heading"><h2>Continuation</h2><span id="autocomplete-tokens" class="metrics"></span></div>
        <pre id="autocomplete-suggestion" class="continuation-output" aria-label="Generated continuation"></pre>
        <p id="autocomplete-metrics" class="metrics">No continuation generated.</p>
        <div class="editor-toolbar"><span id="autocomplete-insertion" class="field-help">Accept inserts at the cursor position used for generation.</span><div class="button-row"><button type="button" id="autocomplete-copy" class="text-button">Copy continuation</button><button type="button" id="autocomplete-discard" class="text-button">Discard</button><button type="button" id="autocomplete-accept" class="primary-button">Accept continuation</button></div></div>
      </section></div><aside id="autocomplete-information" class="panel run-information" aria-label="Selected model information"></aside></div>`;
    this.form = $('#autocomplete-form', root);
    this.editor = $('#autocomplete-text', root);
    this.select = $('#autocomplete-checkpoint', root);
    $('.sampling', root).open = true;
    wireSettings(root);
    this.saved = new SavedInputs(app, item => this.restore(item));
    this.form.addEventListener('submit', event => { event.preventDefault(); this.complete(); });
    this.select.addEventListener('change', () => { this.id = null; this.discard(); this.renderInformation(); this.updateBusy(); });
    this.editor.addEventListener('input', () => { this.discard(); this.updatePosition(); this.updateBusy(); });
    for (const event of ['click', 'keyup', 'select']) this.editor.addEventListener(event, () => { this.updatePosition(); this.updateBusy(); });
    this.editor.addEventListener('keydown', event => {
      if (event.isComposing) return;
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        if (!$('#autocomplete-generate', root).disabled) this.form.requestSubmit();
      } else if (event.key === 'Tab' && !event.shiftKey && this.suggestion && !this.controller) {
        event.preventDefault(); this.accept();
      } else if (event.key === 'Escape' && this.suggestion && !this.controller) {
        event.preventDefault(); this.discard();
      }
    });
    $('#autocomplete-cancel', root).addEventListener('click', () => this.controller?.abort());
    $('#autocomplete-accept', root).addEventListener('click', () => this.accept());
    $('#autocomplete-discard', root).addEventListener('click', () => this.discard());
    $('#autocomplete-copy', root).addEventListener('click', () => copyText(this.suggestion, text => app.notify(text)));
    $('#clear-history').addEventListener('click', () => {
      if (this.controller) return app.notify('Stop generation before clearing saved inputs.');
      this.saved.clear(); this.id = null; app.notify('Saved inputs cleared');
    });
    this.updateBusy();
  }

  refresh() {
    const previous = this.select.value;
    populateCheckpoints(this.select, this.app.runs);
    if (!previous && !this.editor.value) {
      const run = this.app.runs.find(item => item.checkpoints?.length);
      if (run) this.select.value = refKey({ run_id: run.id, checkpoint_id: run.default_checkpoint || run.checkpoints[0].id });
    }
    this.saved.render(); this.renderInformation(); this.updateBusy();
  }

  renderInformation() {
    const ref = parseRef(this.select.value);
    const run = this.app.runs.find(item => item.id === ref?.run_id);
    const checkpoint = run?.checkpoints?.find(item => item.id === ref?.checkpoint_id);
    renderRunInformation($('#autocomplete-information', this.root), run, checkpoint, this.app.catalog?.device);
  }

  updatePosition() {
    const start = this.editor.selectionStart;
    $('#autocomplete-position', this.root).textContent = `${formatNumber(this.editor.value.length, 0)} characters · cursor ${formatNumber(start, 0)}${this.editor.selectionEnd > start ? ` · ${this.editor.selectionEnd - start} selected` : ''}`;
  }

  updateBusy() {
    const running = Boolean(this.controller);
    const hasContext = this.editor.selectionStart > 0;
    $('#autocomplete-generate', this.root).disabled = Boolean(this.app.busy) || !this.select.value || !hasContext;
    $('#autocomplete-cancel', this.root).hidden = !running;
    this.editor.disabled = running; this.select.disabled = running;
    this.root.querySelectorAll('.settings-grid input').forEach(input => { input.disabled = running; });
    for (const action of ['accept', 'discard', 'copy']) $('#autocomplete-' + action, this.root).disabled = running || !this.suggestion;
    const status = $('#autocomplete-status', this.root);
    status.textContent = running ? 'Generating…' : this.app.busy ? `${this.app.busy.kind === 'training' ? 'Training' : 'Generation'} is using the compute device.` : !this.select.value ? 'No checkpoint available. Use Train to create a model.' : !hasContext ? 'Enter text and place the cursor after the context.' : 'Ready';
  }

  discard() {
    if (this.controller) return;
    this.suggestion = ''; this.metrics = null; this.replacement = null;
    $('#autocomplete-suggestion', this.root).textContent = '';
    $('#autocomplete-metrics', this.root).textContent = 'No continuation generated.';
    $('#autocomplete-tokens', this.root).textContent = '';
    $('#autocomplete-error', this.root).hidden = true;
    this.updateBusy();
  }

  save() {
    const ref = parseRef(this.select.value);
    if (!ref || !this.editor.value) return;
    this.id = this.saved.remember({ id: this.id, ref, text: this.editor.value,
      start: this.replacement?.start ?? this.editor.selectionStart,
      end: this.replacement?.end ?? this.editor.selectionEnd,
      suggestion: this.suggestion, metrics: this.metrics, settings: readSettings(this.root) });
  }

  restore(item) {
    if (this.controller) return this.app.notify('Stop generation before switching inputs.');
    this.save();
    this.discard(); this.id = item.id;
    this.editor.value = item.text; this.select.value = refKey(item.ref);
    this.editor.setSelectionRange(item.start ?? item.text.length, item.end ?? item.text.length);
    this.suggestion = item.suggestion || ''; this.metrics = item.metrics || null;
    this.replacement = this.suggestion ? { text: item.text, start: this.editor.selectionStart, end: this.editor.selectionEnd } : null;
    restoreSettings(this.root, item.settings);
    $('#autocomplete-suggestion', this.root).textContent = this.suggestion;
    $('#autocomplete-metrics', this.root).textContent = this.metrics ? metricsText(this.metrics) : 'No continuation generated.';
    this.updatePosition(); this.renderInformation(); this.updateBusy();
    this.app.navigate('autocomplete');
    if (!this.select.value) this.showError('The saved checkpoint is no longer available. Select another checkpoint to generate.');
  }

  newInput() {
    if (this.controller) return this.app.notify('Stop generation before starting a new input.');
    this.save(); this.discard(); this.id = null; this.editor.value = '';
    this.updatePosition(); this.updateBusy(); this.editor.focus();
  }

  accept() {
    if (this.controller || !this.suggestion || !this.replacement) return;
    const { text, start, end } = this.replacement;
    if (this.editor.value !== text) { this.discard(); return; }
    const cursor = start + this.suggestion.length;
    this.editor.value = text.slice(0, start) + this.suggestion + text.slice(end);
    this.editor.setSelectionRange(cursor, cursor);
    this.discard(); this.updatePosition(); this.save(); this.editor.focus();
  }

  showError(message) { $('#autocomplete-error', this.root).textContent = message; $('#autocomplete-error', this.root).hidden = false; }

  async complete() {
    const ref = parseRef(this.select.value);
    const text = this.editor.value;
    const start = this.editor.selectionStart, end = this.editor.selectionEnd;
    if (!ref || !start || this.app.busy || !this.form.reportValidity()) return;
    this.discard(); this.replacement = { text, start, end };
    this.controller = new AbortController();
    this.app.setLocalOperation({ kind: 'generation', id: 'autocomplete' });
    try {
      await generate({ models: [ref], prompt: text.slice(0, start), ...readSettings(this.root) }, (event, data) => {
        if (event === 'output' || event === 'result') {
          this.suggestion = data.text || '';
          $('#autocomplete-suggestion', this.root).textContent = this.suggestion;
          $('#autocomplete-tokens', this.root).textContent = `${data.generated_tokens || 0} generated tokens`;
          if (event === 'result') this.metrics = data;
          $('#autocomplete-metrics', this.root).textContent = event === 'result' ? metricsText(data) : 'Generating…';
        } else if (event === 'error') throw new Error(data.message || 'Generation failed.');
      }, this.controller.signal);
    } catch (error) {
      if (error.name === 'AbortError') $('#autocomplete-metrics', this.root).textContent = 'Stopped. Partial continuation retained.';
      else { this.showError(error.message); $('#autocomplete-metrics', this.root).textContent = 'Generation failed.'; }
    } finally {
      this.controller = null; this.save(); this.app.setLocalOperation(null);
      await this.app.pollStatus(); this.updateBusy(); this.editor.focus();
    }
  }
}

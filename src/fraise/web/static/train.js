import { post } from '/static/api.js';
import { $, $$, element, option, populateCheckpoints, parseRef, refKey, formatNumber } from '/static/ui.js';
import { TrainingMonitor } from '/static/training-monitor.js';

const COMMON_TRAINING = ['batch_size', 'minibatch_size', 'num_epochs', 'sequence_length', 'max_train_tokens', 'learning_rate', 'adamw_learning_rate', 'use_bf16', 'use_compile', 'checkpoint_interval_steps', 'log_interval_steps'];
const LABELS = { residual_stream_dim: 'Residual dimension', head_dim: 'Head dimension', qk_dim: 'Query / key dimension', num_heads: 'Attention heads', swiglu_up_dim: 'SwiGLU dimension', num_layers: 'Layers', max_seq_len: 'Context length', tie_word_embeddings: 'Tie word embeddings', vocab_size: 'Vocabulary size', batch_size: 'Batch size', minibatch_size: 'Minibatch size', num_epochs: 'Epochs', sequence_length: 'Sequence length', max_train_tokens: 'Training tokens / epoch', learning_rate: 'Muon learning rate', adamw_learning_rate: 'AdamW learning rate', use_bf16: 'Use bfloat16', use_compile: 'Compile model', checkpoint_interval_steps: 'Checkpoint every (steps)', log_interval_steps: 'Log every (steps)' };
const labelFor = name => LABELS[name] || name.replaceAll('_', ' ').replace(/^./, character => character.toUpperCase());

export class Train {
  constructor(root, app) {
    this.root = root; this.app = app;
    this.model = {}; this.training = {}; this.mode = 'fresh';
    this.initialized = false; this.requesting = false;
    root.innerHTML = `<div class="page-heading"><div><h1>Train model</h1><p>Configure a fresh run or resume a saved training checkpoint.</p></div></div>
      <div class="train-layout"><div class="train-config"><div class="train-controls"><div class="segmented" role="group" aria-label="Training mode"><button type="button" data-train-mode="fresh" class="active">Fresh run</button><button type="button" data-train-mode="resume">Resume run</button></div><span class="subtle" id="train-device">Runs on your local machine</span></div>
      <form id="train-form"><div class="panel"><div class="panel-heading"><h2>Run configuration</h2></div><div class="form-grid two-columns"><label class="field-wide" id="train-resume-label" hidden>Resume from checkpoint<select id="train-resume"><option value="">Choose a saved checkpoint</option></select></label><label class="field-wide" id="train-preset-label">Preset or saved configuration<select id="train-preset"><option value="">Loading presets…</option></select></label><label class="field-wide">Run name<input id="train-name" required maxlength="100" pattern="[A-Za-z0-9][A-Za-z0-9._-]*" placeholder="run-001" title="Start with a letter or number; use letters, numbers, dots, dashes, and underscores."></label><label>Architecture<select id="train-architecture" required></select></label><label>Tokenizer<select id="train-tokenizer" required><option value="">Select a tokenizer</option></select></label></div><p class="field-help" id="train-tokenizer-help">The tokenizer sets your model’s vocabulary size.</p><div class="form-notice" id="train-resume-notice" hidden>Resuming preserves the saved model, tokenizer, and training identity. Only operational settings can change.</div><p class="inline-error" id="train-setup-error" role="alert" hidden></p></div>
      <div class="panel"><div class="panel-heading"><h2>Model configuration</h2><span class="badge" id="train-parameters">Configure a model</span></div><div class="form-grid" id="train-model-fields"></div><details class="advanced-settings"><summary>Full model configuration</summary><label class="json-label" for="train-model-json">Model configuration · JSON</label><textarea class="json-editor" id="train-model-json" spellcheck="false" aria-describedby="train-json-help"></textarea></details></div>
      <div class="panel"><div class="panel-heading"><h2>Training settings</h2></div><div class="form-grid" id="train-training-fields"></div><hr class="divider"><h3 class="section-label">Dataset token budgets</h3><div class="source-grid" id="train-sources"></div><div class="token-total" id="train-token-total"></div><details class="advanced-settings"><summary>Full training configuration</summary><p class="field-help" id="train-json-help">Common controls and JSON edit the same configuration. Use JSON for optimizer, validation, packing, and other advanced settings.</p><label class="json-label" for="train-training-json">Training configuration · JSON</label><textarea class="json-editor" id="train-training-json" spellcheck="false"></textarea><p class="field-help" id="train-operational-help" hidden></p></details><p id="train-validation" class="validation-status" role="status" hidden></p><p id="train-error" class="inline-error" role="alert" hidden></p><div class="train-actions"><button type="button" class="secondary-button" id="train-validate">Check configuration</button><button type="submit" class="primary-button" id="train-start">Start training</button></div></div></form></div>
      <div class="train-monitor"><div class="panel"><div class="monitor-header"><h2>Training status</h2><span class="badge" id="job-status">Ready</span></div><p class="monitor-phase" id="job-phase">Idle</p><p class="monitor-caption" id="job-caption">Start or resume a run to view progress, loss, logs, and checkpoints.</p><div class="progress-track" role="progressbar" aria-label="Training progress" aria-valuemin="0" aria-valuemax="100" id="job-progress"><div class="progress-fill" id="job-progress-fill"></div></div><div class="progress-caption"><span id="job-progress-label">Waiting for a run</span><span id="job-progress-percent">—</span></div><div class="stat-grid"><div><p class="stat-label">Global step</p><div class="stat-value" id="job-step">—</div></div><div><p class="stat-label">Latest loss</p><div class="stat-value" id="job-loss">—</div></div><div><p class="stat-label">Tokens seen</p><div class="stat-value" id="job-tokens">—</div></div><div><p class="stat-label">Tokens / second</p><div class="stat-value" id="job-speed">—</div></div></div><div class="loss-chart"><div class="loss-chart-heading"><span>Training loss</span><span>by global step</span></div><svg id="job-chart" viewBox="0 0 400 150" role="img" aria-label="Training loss chart, waiting for data"></svg></div><p class="checkpoint-notice" id="job-checkpoint" hidden></p><p class="inline-error" id="job-error" role="alert" hidden></p><button type="button" class="danger-button monitor-stop" id="job-stop" hidden>Stop & save after current step</button><p class="field-help" id="job-stop-note" hidden>Stopping safely can take a moment. The current step finishes before the checkpoint is saved.</p><details class="log-panel"><summary>Training logs</summary><pre class="log-output" id="job-logs">No logs yet.</pre></details></div></div></div>`;
    this.form = $('#train-form', root);
    $$('[data-train-mode]', root).forEach(button => button.addEventListener('click', () => this.changeMode(button.dataset.trainMode)));
    $('#train-resume', root).addEventListener('change', () => this.loadResume());
    $('#train-preset', root).addEventListener('change', () => this.loadPreset());
    $('#train-architecture', root).addEventListener('change', () => this.loadArchitecture());
    $('#train-tokenizer', root).addEventListener('change', () => { this.applyTokenizer(); this.renderFields(); this.syncEditors(); this.invalidate(); });
    $('#train-name', root).addEventListener('input', () => this.invalidate());
    $('#train-model-json', root).addEventListener('input', () => this.readEditor('model'));
    $('#train-training-json', root).addEventListener('input', () => this.readEditor('training'));
    $('#train-validate', root).addEventListener('click', () => this.validateOnly());
    this.form.addEventListener('submit', event => { event.preventDefault(); this.start(); });
    this.monitor = new TrainingMonitor(root, app);
  }

  refresh() {
    const catalog = this.app.catalog;
    if (!catalog) return;
    if (!this.initialized) {
      this.model = structuredClone(catalog.model_defaults || {});
      const architecture = catalog.architectures?.find(item => item.id === this.model.architecture) || catalog.architectures?.[0];
      if (architecture) this.model = { ...this.model, ...structuredClone(architecture.example), architecture: architecture.id };
      this.training = structuredClone(catalog.training_defaults || {});
      this.initialized = true;
      $('#train-name', this.root).value = this.uniqueName('run-001');
    }
    const preset = $('#train-preset', this.root);
    const previousPreset = preset.value;
    preset.replaceChildren();
    for (const architecture of catalog.architectures || []) preset.append(option(`preset:${architecture.id}`, `${architecture.label} preset`));
    const clones = document.createElement('optgroup'); clones.label = 'Clone a saved configuration';
    for (const run of this.app.runs.filter(run => run.config && run.training)) clones.append(option(`clone:${run.id}`, run.name || run.id));
    if (clones.children.length) preset.append(clones);
    if ([...preset.options].some(item => item.value === previousPreset)) preset.value = previousPreset;
    const architectureSelect = $('#train-architecture', this.root);
    architectureSelect.replaceChildren(...(catalog.architectures || []).map(architecture => option(architecture.id, architecture.label)));
    architectureSelect.value = this.model.architecture || catalog.architectures?.[0]?.id || '';
    const tokenizerSelect = $('#train-tokenizer', this.root);
    const tokenizer = tokenizerSelect.value;
    tokenizerSelect.replaceChildren(option('', 'Select a tokenizer'), ...(catalog.tokenizers || []).map(item => option(item.id, `${item.label} · ${formatNumber(item.vocab_size, 0)}`)));
    tokenizerSelect.value = tokenizer || catalog.tokenizers?.[0]?.id || '';
    const resumable = this.app.runs.filter(run => run.can_resume).map(run => ({ ...run, checkpoints: run.checkpoints.filter(checkpoint => checkpoint.kind === 'checkpoint') }));
    populateCheckpoints($('#train-resume', this.root), resumable, undefined, 'Choose a saved checkpoint');
    if (this.mode === 'resume') this.loadResume(false); else { this.applyTokenizer(); this.renderFields(); this.syncEditors(); }
    const device = catalog.device;
    $('#train-device', this.root).textContent = device?.resolved ? `Device: ${String(device.resolved).toUpperCase()}` : 'Runs on your local machine';
    this.updateBusy();
  }

  uniqueName(base) {
    const names = new Set(this.app.runs.map(run => run.id));
    let name = base; let index = 2;
    while (names.has(name)) name = `${base}-${index++}`;
    return name;
  }

  invalidate() { $('#train-validation', this.root).hidden = true; $('#train-parameters', this.root).textContent = 'Not yet checked'; $('#train-error', this.root).hidden = true; }

  applyTokenizer() {
    const tokenizer = this.app.catalog.tokenizers?.find(item => item.id === $('#train-tokenizer', this.root).value);
    if (tokenizer && this.mode === 'fresh') this.model.vocab_size = tokenizer.vocab_size;
    $('#train-tokenizer-help', this.root).textContent = tokenizer ? `${formatNumber(tokenizer.vocab_size, 0)} vocabulary tokens. The selected tokenizer is saved with the run.` : 'No tokenizer is available. Create one with Fraise’s create-tokenizer command, then refresh the page.';
    $('#train-setup-error', this.root).hidden = true;
  }

  loadArchitecture() {
    const architecture = this.app.catalog.architectures.find(item => item.id === $('#train-architecture', this.root).value);
    if (!architecture) return;
    this.model = { ...structuredClone(architecture.example), architecture: architecture.id };
    this.applyTokenizer(); this.renderFields(); this.syncEditors(); this.invalidate();
  }

  loadPreset() {
    const value = $('#train-preset', this.root).value;
    if (value.startsWith('clone:')) {
      const run = this.app.runs.find(run => run.id === value.slice(6));
      if (!run) return;
      this.model = structuredClone(run.config); this.training = structuredClone(run.training);
      $('#train-name', this.root).value = this.uniqueName(`${run.id}-copy`);
      $('#train-tokenizer', this.root).value = run.tokenizer?.id || '';
      $('#train-architecture', this.root).value = this.model.architecture;
    } else {
      $('#train-architecture', this.root).value = value.slice(7);
      this.training = structuredClone(this.app.catalog.training_defaults || {});
      this.loadArchitecture();
    }
    this.applyTokenizer(); this.renderFields(); this.syncEditors(); this.invalidate(); this.updateBusy();
  }

  changeMode(mode) {
    if (this.requesting || this.app.busy || !this.initialized || mode === this.mode) return;
    if (mode === 'resume') this.freshDraft = { model: structuredClone(this.model), training: structuredClone(this.training), name: $('#train-name', this.root).value, tokenizer: $('#train-tokenizer', this.root).value };
    this.mode = mode;
    $$('[data-train-mode]', this.root).forEach(button => { button.classList.toggle('active', button.dataset.trainMode === mode); button.setAttribute('aria-pressed', String(button.dataset.trainMode === mode)); });
    $('#train-resume-label', this.root).hidden = mode !== 'resume';
    $('#train-preset-label', this.root).hidden = mode === 'resume';
    $('#train-resume-notice', this.root).hidden = mode !== 'resume';
    $('#train-operational-help', this.root).hidden = mode !== 'resume';
    $('#train-operational-help', this.root).textContent = `Editable operational settings: ${(this.app.catalog.operational_settings || []).join(', ')}.`;
    if (mode === 'resume') this.loadResume();
    else if (this.freshDraft) {
      this.model = this.freshDraft.model; this.training = this.freshDraft.training;
      $('#train-name', this.root).value = this.freshDraft.name;
      $('#train-tokenizer', this.root).value = this.freshDraft.tokenizer;
      $('#train-architecture', this.root).value = this.model.architecture;
      this.applyTokenizer(); this.renderFields(); this.syncEditors();
    }
    this.invalidate(); this.updateBusy();
  }

  loadResume(reset = true) {
    const ref = parseRef($('#train-resume', this.root).value);
    const run = this.app.runs.find(run => run.id === ref?.run_id);
    this.resumeRun = run || null;
    if (run) {
      if (reset || !this.resumeIdentity || this.resumeIdentity.run_id !== run.id) {
        this.model = structuredClone(run.config); this.training = structuredClone(run.training);
      }
      this.resumeIdentity = ref;
      $('#train-name', this.root).value = run.name || run.id;
      $('#train-architecture', this.root).value = run.architecture;
      $('#train-tokenizer', this.root).value = run.tokenizer?.id || '';
      this.applyTokenizer();
    } else {
      this.resumeIdentity = null;
      $('#train-setup-error', this.root).textContent = this.app.runs.some(run => run.can_resume) ? 'Choose a training checkpoint to resume.' : 'No resumable checkpoints are available. Start a fresh run or migrate an older saved run first.';
      $('#train-setup-error', this.root).hidden = false;
    }
    this.renderFields(); this.syncEditors(); this.invalidate(); this.updateBusy();
  }

  makeField(name, value, section, type) {
    const label = element('label', typeof value === 'boolean' || type === 'bool' ? 'check-label' : '');
    const input = document.createElement('input');
    input.dataset.section = section; input.dataset.field = name;
    const isBoolean = typeof value === 'boolean' || type === 'bool';
    if (isBoolean) { input.type = 'checkbox'; input.checked = Boolean(value); label.append(input, element('span', '', labelFor(name))); }
    else {
      label.append(document.createTextNode(labelFor(name)));
      input.type = typeof value === 'number' || /^(int|float|number)/.test(type || '') || value == null ? 'number' : 'text';
      input.value = value ?? '';
      if (input.type === 'number') { input.step = Number.isInteger(value) && !name.includes('learning_rate') ? '1' : 'any'; input.min = '0'; }
      label.append(input);
    }
    input.addEventListener('input', () => {
      this[section][name] = isBoolean ? input.checked : input.type === 'number' ? input.value === '' ? null : Number(input.value) : input.value;
      this.syncEditors(); this.invalidate(); this.renderTokenTotal();
    });
    return label;
  }

  renderFields() {
    const architecture = this.app.catalog.architectures?.find(item => item.id === this.model.architecture);
    const modelFields = $('#train-model-fields', this.root); modelFields.replaceChildren();
    const fields = architecture?.fields?.length ? architecture.fields : Object.keys(this.model).filter(name => name !== 'architecture').map(name => ({ name }));
    for (const field of fields) {
      if (field.name === 'architecture') continue;
      const value = this.model[field.name] ?? field.default;
      if (['object', 'array'].includes(field.type) || value !== null && typeof value === 'object') continue;
      modelFields.append(this.makeField(field.name, value, 'model', field.type));
    }
    const trainingFields = $('#train-training-fields', this.root); trainingFields.replaceChildren();
    for (const name of COMMON_TRAINING) if (name in this.training) trainingFields.append(this.makeField(name, this.training[name], 'training'));
    const sources = $('#train-sources', this.root); sources.replaceChildren();
    const counts = this.training.dataset_token_counts || {};
    for (const source of new Set([...(this.app.catalog.sources || []), ...Object.keys(counts)])) {
      const label = element('label', '', labelFor(source));
      const input = document.createElement('input');
      input.type = 'number'; input.step = '1'; input.min = '0'; input.value = counts[source] ?? 0; input.dataset.source = source;
      input.addEventListener('input', () => {
        this.training.dataset_token_counts ||= {};
        this.training.dataset_token_counts[source] = input.value === '' ? 0 : Number(input.value);
        this.syncEditors(); this.renderTokenTotal(); this.invalidate();
      });
      label.append(input); sources.append(label);
    }
    this.renderTokenTotal(); this.updateBusy();
  }

  renderTokenTotal() {
    const count = Object.values(this.training.dataset_token_counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
    const total = $('#train-token-total', this.root);
    total.classList.toggle('invalid', count !== Number(this.training.max_train_tokens));
    total.replaceChildren(element('span', '', `Source total: ${formatNumber(count, 0)}`), element('span', '', count === Number(this.training.max_train_tokens) ? 'Matches training token budget ✓' : `Must equal ${formatNumber(this.training.max_train_tokens, 0)} tokens`));
  }

  syncEditors() {
    for (const section of ['model', 'training']) {
      const editor = $(`#train-${section}-json`, this.root);
      editor.value = JSON.stringify(this[section], null, 2); editor.setCustomValidity('');
    }
  }

  readEditor(section) {
    const editor = $(`#train-${section}-json`, this.root);
    try {
      const value = JSON.parse(editor.value);
      if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Use a JSON object.');
      if (this.mode === 'resume' && section === 'training' && this.resumeRun) {
        const allowed = new Set(this.app.catalog.operational_settings || []);
        const immutable = new Set([...Object.keys(this.resumeRun.training), ...Object.keys(value)]);
        for (const key of immutable) if (!allowed.has(key) && JSON.stringify(value[key]) !== JSON.stringify(this.resumeRun.training[key])) throw new Error(`${key} is fixed when resuming. Only operational settings can change.`);
      }
      this[section] = value; editor.setCustomValidity(''); this.invalidate();
      if (section === 'model') $('#train-architecture', this.root).value = value.architecture || '';
      this.renderFields();
    } catch (error) { editor.setCustomValidity(`Invalid ${section} configuration: ${error.message}`); this.invalidate(); }
  }

  updateBusy() {
    const locked = this.requesting || Boolean(this.app.busy);
    const resume = this.mode === 'resume';
    $$('[data-train-mode]', this.root).forEach(button => { button.disabled = locked || !this.initialized; });
    for (const id of ['train-name', 'train-architecture', 'train-tokenizer', 'train-preset']) $(`#${id}`, this.root).disabled = locked || resume;
    $('#train-resume', this.root).disabled = locked || !resume;
    $$('[data-section]', this.root).forEach(input => {
      input.disabled = locked || input.dataset.field === 'vocab_size' || resume && (input.dataset.section === 'model' || !(this.app.catalog?.operational_settings || []).includes(input.dataset.field));
    });
    $$('[data-source]', this.root).forEach(input => { input.disabled = locked || resume; });
    $('#train-model-json', this.root).disabled = locked || resume;
    $('#train-training-json', this.root).disabled = locked || resume && !this.resumeRun;
    const cannotStart = locked || !this.initialized || !$('#train-tokenizer', this.root).value || resume && !this.resumeRun;
    $('#train-start', this.root).disabled = cannotStart;
    $('#train-validate', this.root).disabled = cannotStart;
    $('#train-start', this.root).textContent = this.requesting ? 'Preparing…' : resume ? 'Resume training' : 'Start training';
    this.monitor?.updateBusy();
  }

  payload() {
    if (!this.form.reportValidity()) return null;
    const name = $('#train-name', this.root).value.trim();
    if (this.mode === 'fresh' && this.app.runs.some(run => run.id === name || run.name === name)) throw new Error('That run name already exists. Choose a unique name, or select Resume run.');
    return { model_config: this.model, training_config: this.training, tokenizer_id: $('#train-tokenizer', this.root).value, run_name: name, resume: this.mode === 'resume' ? parseRef($('#train-resume', this.root).value) : null };
  }

  showError(error) { $('#train-error', this.root).textContent = error.message; $('#train-error', this.root).hidden = false; }

  async validate(payload) {
    const result = await post('/api/config/validate', payload);
    this.model = result.model_config; this.training = result.training_config;
    this.renderFields(); this.syncEditors();
    $('#train-parameters', this.root).textContent = `${formatNumber(result.parameter_count, 0)} parameters`;
    const notice = $('#train-validation', this.root);
    notice.textContent = `Configuration is ready · ${formatNumber(result.parameter_count, 0)} parameters`;
    notice.hidden = false;
    return { ...payload, model_config: this.model, training_config: this.training };
  }

  async validateOnly() {
    if (this.requesting || this.app.busy) return;
    try {
      const payload = this.payload(); if (!payload) return;
      this.requesting = true; this.updateBusy(); $('#train-error', this.root).hidden = true;
      await this.validate(payload);
    } catch (error) { this.showError(error); }
    finally { this.requesting = false; this.updateBusy(); }
  }

  async start() {
    if (this.requesting || this.app.busy) return;
    try {
      const payload = this.payload(); if (!payload) return;
      this.requesting = true; this.updateBusy(); $('#train-error', this.root).hidden = true;
      const validated = await this.validate(payload);
      const job = await post('/api/jobs', validated);
      this.monitor.begin(job);
      this.app.notify(this.mode === 'resume' ? 'Training resumed' : 'Training started');
    } catch (error) { this.showError(error); }
    finally { this.requesting = false; this.updateBusy(); }
  }

  async restoreJob() { await this.monitor.restoreJob(); }
}

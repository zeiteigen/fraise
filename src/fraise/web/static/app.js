import { request } from '/static/api.js';
import { $, $$, element } from '/static/ui.js';
import { Autocomplete } from '/static/autocomplete.js';
import { Compare } from '/static/compare.js';
import { Train } from '/static/train.js';
import { Tokens } from '/static/tokens.js';

class Workspace {
  constructor() {
    this.catalog = null;
    this.runs = [];
    this.serverBusy = null;
    this.localOperation = null;
    this.views = {};
    this.view = 'autocomplete';
    this.toastTimer = null;
    this.views.autocomplete = new Autocomplete($('#autocomplete-view'), this);
    this.views.compare = new Compare($('#compare-view'), this);
    this.views.train = new Train($('#train-view'), this);
    this.views.tokens = new Tokens($('#tokens-view'), this);
    $$('[data-view]').forEach(button => button.addEventListener('click', () => this.navigate(button.dataset.view)));
    $('#new-input').addEventListener('click', () => { this.views.autocomplete.newInput(); this.navigate('autocomplete'); });
    $('#menu-toggle').addEventListener('click', () => this.toggleSidebar());
    $('#sidebar-shade').addEventListener('click', () => this.toggleSidebar(false));
    document.addEventListener('keydown', event => { if (event.key === 'Escape') this.toggleSidebar(false); });
    window.addEventListener('hashchange', () => this.navigate(location.hash.slice(1), false));
    this.navigate(location.hash.slice(1) || 'autocomplete', false);
    this.start();
  }

  get busy() { return this.localOperation || this.serverBusy; }

  navigate(view, updateHash = true) {
    if (!this.views[view]) view = 'autocomplete';
    this.view = view;
    for (const [name, instance] of Object.entries(this.views)) instance.root.hidden = name !== view;
    $$('[data-view]').forEach(button => {
      const active = button.dataset.view === view;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    });
    $('#view-label').textContent = { autocomplete: 'Autocomplete', compare: 'Compare', train: 'Train', tokens: 'Tokenization' }[view];
    $('#view-subtitle').textContent = { autocomplete: 'Base language model evaluation', compare: 'Shared input and sampling settings', train: 'Configuration, progress, and checkpoints', tokens: 'Segmentation and hierarchical composition' }[view];
    if (updateHash) history.replaceState(null, '', `#${view}`);
    this.toggleSidebar(false);
  }

  toggleSidebar(open = !document.body.classList.contains('sidebar-open')) {
    document.body.classList.toggle('sidebar-open', open);
    $('#sidebar-shade').hidden = !open;
    $('#menu-toggle').setAttribute('aria-expanded', String(open));
  }

  notify(message) {
    clearTimeout(this.toastTimer);
    $('#toast').textContent = message;
    $('#toast').hidden = false;
    this.toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 3500);
  }

  setLocalOperation(operation) { this.localOperation = operation; this.updateBusy(); }

  updateBusy() {
    const busy = this.busy;
    const status = $('#global-status');
    status.classList.toggle('is-busy', Boolean(busy));
    status.lastElementChild.textContent = busy ? busy.kind === 'training' ? 'Training in progress' : 'Generating' : 'Local workspace';
    Object.values(this.views).forEach(view => view.updateBusy());
  }

  async refreshRuns() {
    const [data, catalog] = await Promise.all([request('/api/runs'), request('/api/catalog')]);
    this.runs = data.runs || [];
    this.catalog = catalog;
    Object.values(this.views).forEach(view => view.refresh());
  }

  async start() {
    try {
      const results = await Promise.all([request('/api/catalog'), request('/api/runs')]);
      this.catalog = results[0];
      this.runs = results[1].runs || [];
      const device = this.catalog.device;
      $('#device-label').textContent = device?.resolved ? `${String(device.resolved).toUpperCase()} workspace` : 'Local workspace';
      $('#device-dot').classList.add('connected');
      Object.values(this.views).forEach(view => view.refresh());
      await this.views.train.restoreJob();
      await this.pollStatus();
    } catch (error) {
      this.connectionError(error);
      const retry = element('button', 'text-button', 'Reconnect');
      retry.addEventListener('click', () => { $('#connection-error').hidden = true; this.start(); }, { once: true });
      $('#connection-error').append(' ', retry);
    }
    if (!this.statusTimer) this.statusTimer = setInterval(() => this.pollStatus(), 2000);
  }

  connectionError(error) {
    const notice = $('#connection-error');
    notice.textContent = `Could not connect to the local workspace. ${error.message}`;
    notice.hidden = false;
  }

  async pollStatus() {
    if (this.pollingStatus) return;
    this.pollingStatus = true;
    try {
      const status = await request('/api/status');
      const wasBusy = this.serverBusy;
      this.serverBusy = status.busy;
      if (!this.catalog) {
        await this.refreshRuns();
        const device = this.catalog.device;
        $('#device-label').textContent = device?.resolved ? `${String(device.resolved).toUpperCase()} workspace` : 'Local workspace';
        $('#device-dot').classList.add('connected');
        await this.views.train.restoreJob();
      }
      this.updateBusy();
      if (this.catalog) $('#connection-error').hidden = true;
      if (wasBusy && !this.serverBusy && this.catalog) await this.refreshRuns();
    } catch (error) { this.connectionError(error); }
    finally { this.pollingStatus = false; }
  }
}

new Workspace();

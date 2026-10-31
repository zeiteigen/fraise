import { post, request } from '/static/api.js';
import { $, formatNumber } from '/static/ui.js';
import { renderLossChart } from '/static/training-chart.js';

const ACTIVE = new Set(['starting', 'running', 'stopping']);
const labelFor = name => name.replaceAll('_', ' ').replace(/^./, character => character.toUpperCase());

export class TrainingMonitor {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.job = null;
    this.cursor = 0;
    this.logs = '';
    this.points = new Map();
    this.progress = {};
    $('#job-stop', root).addEventListener('click', () => this.stop());
    renderLossChart($('#job-chart', root), []);
  }

  updateBusy() {
    $('#job-stop', this.root).hidden = !this.job || !ACTIVE.has(this.job.status);
    $('#job-stop', this.root).disabled = this.job?.status === 'stopping' || this.stopping;
  }

  async restoreJob() {
    try {
      const response = await request('/api/jobs');
      const jobs = (response.jobs || []).sort((a, b) => Number(a.created_at) - Number(b.created_at));
      const job = jobs.find(job => ACTIVE.has(job.status)) || jobs.at(-1);
      if (job) this.begin(job);
    } catch (error) { $('#job-caption', this.root).textContent = `Could not load training jobs. ${error.message}`; }
  }

  begin(job) {
    if (this.pollTimer) clearInterval(this.pollTimer);
    this.job = job; this.cursor = 0; this.logs = ''; this.points.clear(); this.progress = {};
    this.stopping = false;
    $('#job-error', this.root).hidden = true; $('#job-checkpoint', this.root).hidden = true;
    $('#job-stop-note', this.root).hidden = true;
    if (ACTIVE.has(job.status)) this.app.setLocalOperation({ kind: 'training', id: job.id });
    this.pollJob();
    this.pollTimer = setInterval(() => this.pollJob(), 1000);
    this.updateBusy();
  }

  async pollJob() {
    if (!this.job || this.polling) return;
    this.polling = true;
    try {
      const data = await request(`/api/jobs/${encodeURIComponent(this.job.id)}?cursor=${this.cursor}`);
      this.job = data;
      this.cursor = data.cursor ?? this.cursor;
      if (data.logs) this.logs = (this.logs + data.logs).slice(-200000);
      for (const event of data.events || []) {
        if (event.type === 'progress') {
          this.progress = { ...this.progress, ...event };
          if (Number.isFinite(event.loss) && event.global_step != null) this.points.set(event.global_step, { step: event.global_step, loss: event.loss });
          if (this.points.size > 1000) this.points.delete(this.points.keys().next().value);
        }
      }
      this.renderJob();
      if (!ACTIVE.has(data.status)) {
        clearInterval(this.pollTimer); this.pollTimer = null;
        if (this.app.localOperation?.id === data.id) this.app.setLocalOperation(null);
        await this.app.refreshRuns();
        await this.app.pollStatus();
      }
    } catch (error) {
      $('#job-error', this.root).textContent = `Connection interrupted. Retrying… ${error.message}`;
      $('#job-error', this.root).hidden = false;
    } finally { this.polling = false; }
  }

  renderJob() {
    const job = this.job; const progress = this.progress;
    const status = $('#job-status', this.root);
    status.textContent = labelFor(job.status);
    status.className = `badge${job.status === 'completed' ? ' success' : job.status === 'failed' ? ' failed' : ACTIVE.has(job.status) ? ' running' : ''}`;
    const checkpoint = job.latest_checkpoint || [...(job.events || [])].reverse().find(event => ['checkpoint', 'stopped'].includes(event.type) && event.path)?.path;
    const phase = job.status === 'stopping' ? 'Stopping after current step' : job.status === 'completed' ? 'Completed' : job.status === 'stopped' ? checkpoint ? 'Stopped; checkpoint saved' : 'Preparation canceled' : job.status === 'failed' ? 'Failed' : job.status === 'interrupted' ? 'Interrupted' : labelFor(job.phase || 'Starting');
    $('#job-phase', this.root).textContent = phase;
    $('#job-caption', this.root).textContent = `${job.run_id || 'Training run'}${progress.epoch != null ? ` · epoch ${progress.epoch}` : ''}`;
    $('#job-step', this.root).textContent = formatNumber(progress.global_step, 0);
    $('#job-loss', this.root).textContent = progress.loss != null ? formatNumber(progress.loss, 4) : '—';
    $('#job-tokens', this.root).textContent = formatNumber(progress.tokens_seen, 0);
    $('#job-speed', this.root).textContent = formatNumber(progress.tokens_per_second, 1);
    const total = progress.total_tokens;
    const percentage = total > 0 && progress.tokens_seen != null ? Math.min(100, Math.max(0, progress.tokens_seen / total * 100)) : job.status === 'completed' ? 100 : null;
    const bar = $('#job-progress', this.root); const fill = $('#job-progress-fill', this.root);
    fill.classList.toggle('indeterminate', percentage === null && ACTIVE.has(job.status));
    fill.style.width = percentage === null ? '' : `${percentage}%`;
    if (percentage === null) bar.removeAttribute('aria-valuenow'); else bar.setAttribute('aria-valuenow', String(percentage));
    $('#job-progress-percent', this.root).textContent = percentage === null ? '—' : `${formatNumber(percentage, 1)}%`;
    $('#job-progress-label', this.root).textContent = total ? `${formatNumber(progress.tokens_seen || 0, 0)} / ${formatNumber(total, 0)} tokens` : ACTIVE.has(job.status) ? labelFor(job.phase || 'Preparing') : labelFor(job.status);
    renderLossChart($('#job-chart', this.root), [...this.points.values()].sort((a, b) => a.step - b.step));
    const log = $('#job-logs', this.root);
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 50;
    log.textContent = this.logs || 'No logs yet.';
    if (nearBottom) log.scrollTop = log.scrollHeight;
    $('#job-checkpoint', this.root).hidden = !checkpoint;
    $('#job-checkpoint', this.root).textContent = checkpoint ? `Checkpoint saved · ${checkpoint}` : '';
    $('#job-error', this.root).hidden = !job.error;
    if (job.error) $('#job-error', this.root).textContent = job.error;
    $('#job-stop-note', this.root).hidden = job.status !== 'stopping';
    this.updateBusy();
  }

  async stop() {
    if (!this.job || !ACTIVE.has(this.job.status) || this.stopping) return;
    this.stopping = true; this.updateBusy();
    try {
      this.job = await post(`/api/jobs/${encodeURIComponent(this.job.id)}/stop`, {});
      this.renderJob();
      $('#job-stop-note', this.root).hidden = false;
      this.app.notify('Stop requested. Saving after the current step.');
    } catch (error) { $('#job-error', this.root).textContent = error.message; $('#job-error', this.root).hidden = false; this.stopping = false; this.updateBusy(); }
  }
}

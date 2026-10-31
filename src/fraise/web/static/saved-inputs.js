import { $, element } from '/static/ui.js';

const STORAGE_KEY = 'fraise.saved-inputs.v1';

export class SavedInputs {
  constructor(app, restore) {
    this.app = app;
    this.restore = restore;
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      this.items = Array.isArray(data) ? data.filter(item => item.id && typeof item.text === 'string' && item.ref?.run_id).slice(0, 40) : [];
    } catch { this.items = []; }
    this.render();
  }

  remember(snapshot) {
    const item = { ...snapshot, id: snapshot.id || crypto.randomUUID(), updated: Date.now() };
    this.items = [item, ...this.items.filter(previous => previous.id !== item.id)].slice(0, 40);
    this.persist();
    return item.id;
  }

  clear() { this.items = []; this.persist(); }

  persist() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.items)); }
    catch { this.app.notify('Browser storage is full. The current input remains in the editor.'); }
    this.render();
  }

  render() {
    const list = $('#history-list');
    list.replaceChildren();
    if (!this.items.length) list.append(element('p', 'sidebar-empty', 'No saved inputs.'));
    for (const item of this.items) {
      const button = element('button', 'history-item');
      button.append(element('span', 'history-title', item.text.slice(0, 60) || 'Untitled input'));
      const run = this.app.runs.find(run => run.id === item.ref.run_id);
      button.append(element('small', '', run?.name || item.ref.run_id));
      button.title = item.ref.checkpoint_id || item.ref.run_id;
      button.addEventListener('click', () => this.restore(item));
      list.append(button);
    }
  }
}

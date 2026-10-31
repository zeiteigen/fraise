import { element, formatNumber } from '/static/ui.js';

const numeric = value => value == null ? '—' : formatNumber(value, 0);

export function renderRunInformation(root, run, checkpoint, device) {
  root.replaceChildren(element('h2', '', 'Model information'));
  if (!run) {
    root.append(element('p', 'field-help', 'Select an available checkpoint.'));
    return;
  }
  const validation = Array.isArray(run.validation) ? run.validation[1] : run.validation?.loss;
  const values = [
    ['Run', run.name || run.id], ['Checkpoint', checkpoint?.label || 'Unavailable'],
    ['Architecture', run.architecture], ['Parameters', numeric(run.parameter_count)],
    ['Context limit', `${numeric(run.config?.max_seq_len)} tokens`],
    ['Vocabulary', `${numeric(run.tokenizer?.vocab_size)} tokens`],
    ['Checkpoint step', numeric(checkpoint?.step)], ['Run training tokens', numeric(run.tokens_seen)],
    ['Latest validation loss', validation == null ? '—' : formatNumber(validation, 4)],
    ['Device', device?.resolved?.toUpperCase() || '—'],
  ];
  const description = element('dl', 'info-grid');
  for (const [label, value] of values) {
    const pair = element('div');
    pair.append(element('dt', '', label), element('dd', '', value ?? '—'));
    description.append(pair);
  }
  root.append(description);
  if (run.config) {
    const details = element('details', 'run-details');
    details.append(element('summary', '', 'Model configuration'), element('pre', '', JSON.stringify(run.config, null, 2)));
    root.append(details);
  }
}

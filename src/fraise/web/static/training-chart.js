const SVG_NS = 'http://www.w3.org/2000/svg';

function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderLossChart(svg, points) {
  svg.replaceChildren();
  if (!points.length) {
    svg.append(svgNode('line', { x1: 0, x2: 400, y1: 125, y2: 125, stroke: '#d6d9dd', 'stroke-dasharray': '3 5' }), svgNode('text', { x: 200, y: 72, 'text-anchor': 'middle', class: 'chart-empty' }, 'No loss data'));
    svg.setAttribute('aria-label', 'Training loss chart, waiting for the first training step');
    return;
  }
  const losses = points.map(point => point.loss);
  const low = Math.min(...losses); const high = Math.max(...losses);
  const span = high - low || Math.max(1, Math.abs(high) * 0.1);
  const minimum = low - span * 0.1; const maximum = high + span * 0.1;
  const firstStep = points[0].step; const lastStep = points.at(-1).step;
  const x = step => 35 + (step - firstStep) / (lastStep - firstStep || 1) * 350;
  const y = loss => 115 - (loss - minimum) / (maximum - minimum) * 105;
  for (let index = 0; index < 3; index++) {
    const value = minimum + (maximum - minimum) * index / 2;
    svg.append(svgNode('line', { x1: 35, x2: 385, y1: y(value), y2: y(value), stroke: '#ebe2d1', 'stroke-dasharray': '3 5' }), svgNode('text', { x: 28, y: y(value) + 3, 'text-anchor': 'end', class: 'chart-label' }, value.toFixed(2)));
  }
  const path = points.map((point, index) => `${index ? 'L' : 'M'}${x(point.step).toFixed(2)},${y(point.loss).toFixed(2)}`).join(' ');
  svg.append(svgNode('path', { d: path, fill: 'none', stroke: '#bc644c', 'stroke-width': 1.8, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  const latest = points.at(-1);
  svg.append(svgNode('circle', { cx: x(latest.step), cy: y(latest.loss), r: 3, fill: '#bc644c' }), svgNode('text', { x: 35, y: 139, class: 'chart-label' }, firstStep.toLocaleString()), svgNode('text', { x: 385, y: 139, 'text-anchor': 'end', class: 'chart-label' }, lastStep === firstStep ? '' : lastStep.toLocaleString()));
  svg.setAttribute('aria-label', `Training loss over ${points.length} updates. Latest loss ${latest.loss.toFixed(4)} at step ${latest.step}.`);
}

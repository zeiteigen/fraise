export async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail ?? body?.message;
    throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Request failed (${response.status}).`);
  }
  return response.json();
}

export function post(path, body, options = {}) {
  return request(path, { ...options, method: 'POST', body: JSON.stringify(body) });
}

export async function generate(body, onEvent, signal) {
  const response = await fetch('/api/generate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    const detail = data?.detail;
    throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Generation failed (${response.status}).`);
  }
  if (!response.body) throw new Error('This browser could not open the response stream.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let eventName = 'message';
  let data = [];
  let receivedDone = false;
  function line(value) {
    if (value === '') {
      if (data.length) {
        const parsed = JSON.parse(data.join('\n'));
        if (eventName === 'done') receivedDone = true;
        onEvent(eventName, parsed);
      }
      eventName = 'message';
      data = [];
    } else if (value.startsWith('event:')) eventName = value.slice(6).trim();
    else if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /, ''));
  }
  try {
    while (true) {
      const next = await reader.read();
      buffer += decoder.decode(next.value, { stream: !next.done });
      let end;
      while ((end = buffer.indexOf('\n')) !== -1) {
        line(buffer.slice(0, end).replace(/\r$/, ''));
        buffer = buffer.slice(end + 1);
      }
      if (next.done) break;
    }
    if (buffer) line(buffer.replace(/\r$/, ''));
    line('');
    if (!receivedDone && !signal.aborted) throw new Error('The response stream ended early. Any partial output is kept below.');
  } finally {
    if (!receivedDone) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

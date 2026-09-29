"""Render token offsets and IDs without changing the tokenizer."""

from html import escape


def render_tokenization(text, tokenizer, name="Tokenizer"):
    encoding = tokenizer.encode(text)
    palette = ("#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fed7aa")
    spans, rows = [], []
    for index, (token_id, token, (start, end)) in enumerate(
        zip(encoding.ids, encoding.tokens, encoding.offsets)
    ):
        visible = (
            text[start:end]
            if end > start
            else tokenizer.decode([token_id], skip_special_tokens=False)
        )
        visible = visible.replace(" ", "·").replace("\n", "↵\n").replace("\t", "⇥")
        label = escape(visible)
        spans.append(
            f'<span title="id={token_id}" style="background:{palette[index % len(palette)]};padding:4px;margin:2px;white-space:pre-wrap">{label}</span>'
        )
        rows.append(
            f"<tr><td>{index}</td><td>{token_id}</td><td>{escape(token)}</td><td>{label}</td><td>{start}:{end}</td></tr>"
        )
    count = len(encoding.ids)
    byte_count = len(text.encode("utf-8"))
    return (
        f"<div><h3>{escape(name)}</h3><p>{count} tokens · {len(text)} characters · {byte_count} UTF-8 bytes · "
        f"{byte_count / max(count, 1):.2f} bytes/token</p>{''.join(spans)}"
        "<table><thead><tr><th>#</th><th>ID</th><th>Token</th><th>Input text</th><th>Offsets</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )

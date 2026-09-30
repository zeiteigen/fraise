"""Render packed training examples as standalone HTML."""

from html import escape

from fraise.config.training import TrainingSettings


def render_html(examples, settings: TrainingSettings) -> str:
    """Make the packed windows readable while retaining every tensor value."""
    cards = []
    for example in examples:
        source = escape(example["source"])
        fields = []
        for title, key in (
            ("Input text", "input_text"),
            ("Target text", "target_text"),
        ):
            fields.append(f"<h3>{title}</h3><pre>{escape(example[key])}</pre>")
        for title, key in (
            ("Input IDs", "input_ids"),
            ("Target IDs", "target_ids"),
            ("Target mask (1 = loss target)", "target_mask"),
        ):
            values = example[key]
            if key == "target_mask":
                values = [int(value) for value in values]
            fields.append(
                f"<details><summary>{title}</summary><pre>{escape(str(values))}</pre></details>"
            )
        cards.append(
            f"<article><h2>#{example['number']} · {source}</h2>"
            f"<p>Epoch {example['epoch']} · Batch {example['batch']} · "
            f"Position {example['batch_position']} · "
            f"{example['valid_targets']} valid targets</p>"
            + "".join(fields)
            + "</article>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Fraise training examples</title>"
        "<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;"
        "background:#f7f7f4;color:#222}article{background:white;border:1px solid #ddd;"
        "border-radius:8px;padding:1rem 1.5rem;margin:1.5rem 0}h2{margin:0}h3{margin-bottom:.3rem}"
        "p{color:#555}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f0;"
        "padding:.8rem;border-radius:4px}details{margin:.7rem 0}summary{cursor:pointer}</style>"
        f"<h1>Fraise training examples</h1><p>{len(examples)} packed windows · "
        f"sequence length {settings.sequence_length} · stride {settings.packing_stride} · "
        f"batch size {settings.batch_size} · shuffle per epoch {settings.shuffle_per_epoch} · "
        f"mix datasets {settings.shuffle_data} · seed {settings.seed}. "
        "Input text includes every model input position; target text shows only loss "
        "positions. Expand the ID and mask "
        "sections for the complete tensors, including padding.</p>"
        + "".join(cards)
        + "</html>"
    )

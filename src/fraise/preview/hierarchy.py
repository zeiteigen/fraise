"""Describe how text is segmented and how each input token is composed.

Two different things are reported and must not be confused:

* **Segmentation** is what the tokenizers actually do to the text: the input
  tokenizer chooses global transformer positions and the base tokenizer
  chooses output tokens. Only the final input vocabulary segments text.
* **Composition** is how the embedding of each input token is built through
  the hierarchy's intermediate levels. Intermediate nodes never become
  transformer positions; they are representation layers.

The result is plain JSON for the web visualizer, notebooks, and the CLI.
"""

import numpy as np

from fraise.hierarchy.definition import EmbeddingHierarchy


def analyze_text(
    text: str,
    tokenizer,
    definition: EmbeddingHierarchy | None,
    *,
    add_special_tokens: bool = False,
    name: str = "",
) -> dict:
    base = getattr(tokenizer, "base", tokenizer)
    encoding = base.encode(text, add_special_tokens=add_special_tokens)
    base_ids = list(encoding.ids)
    if hasattr(tokenizer, "encode_base_ids"):
        input_ids = tokenizer.encode_base_ids(base_ids)
        lengths = [int(tokenizer.unit_lengths[token]) for token in input_ids]
    else:
        input_ids, lengths = base_ids, [1] * len(base_ids)
    decode = _decoder(base)
    base_tokens = [
        {
            "index": index,
            "id": token_id,
            "text": decode([token_id]),
            "start": start,
            "end": end,
        }
        for index, (token_id, (start, end)) in enumerate(
            zip(base_ids, encoding.offsets)
        )
    ]
    input_tokens, cursor = [], 0
    for index, (token_id, length) in enumerate(zip(input_ids, lengths)):
        spans = [
            (token["start"], token["end"])
            for token in base_tokens[cursor : cursor + length]
            if token["end"] > token["start"]
        ]
        input_tokens.append(
            {
                "index": index,
                "id": int(token_id),
                "text": decode(base_ids[cursor : cursor + length]),
                "base_start": cursor,
                "base_end": cursor + length,
                "start": spans[0][0] if spans else None,
                "end": spans[-1][1] if spans else None,
            }
        )
        cursor += length
    result = {
        "name": name,
        "text": text,
        "byte_count": len(text.encode("utf-8")),
        "input_vocab_size": tokenizer.get_vocab_size(),
        # Plain tokenizers have no separate base vocabulary: inputs are outputs.
        "macro": hasattr(tokenizer, "encode_base_ids"),
        "base_vocab_size": base.get_vocab_size(),
        "base_tokens": base_tokens,
        "input_tokens": input_tokens,
        "levels": [],
        "rows": [],
        "trees": [],
        "nodes": {},
    }
    if definition is None:
        return result
    return result | _composition(
        definition, input_tokens, decode, tokenizer.get_vocab_size()
    )


def _composition(definition, input_tokens, decode, vocab_size):
    sizes = definition.level_sizes
    top = definition.num_levels - 1
    token_nodes = definition.input_node_ids(vocab_size)
    node_tokens = {int(node): token for token, node in enumerate(token_nodes.tolist())}
    table, counts = definition.atom_expansions()
    nodes: dict[int, dict] = {}

    def describe(node):
        if node not in nodes:
            introduced = definition.introduced_at(node)
            atoms = table[node, : counts[node]].tolist()
            nodes[node] = {
                "node": node,
                "token_id": node_tokens.get(node),
                "text": decode(atoms),
                "atoms": atoms,
                "introduced": introduced,
                "residual_width": int(definition.residual_widths[node]),
                "residual_level": definition.owner_level(node),
                # Only the introducing level composes; higher levels carry it.
                "children": definition.children(node, introduced) if introduced else [],
            }
        return nodes[node]

    def tree(node, level):
        carried = []
        while level > 0 and node < sizes[level - 1]:
            carried.append(level)
            level -= 1
        info = describe(node)
        children = (
            [tree(child, level - 1) for child in definition.children(node, level)]
            if level
            else []
        )
        return {
            "node": node,
            "level": level,
            "carried_through": carried[::-1],
            "children": children,
            "text": info["text"],
        }

    rows = {level: [] for level in range(definition.num_levels)}
    trees = []
    for token in input_tokens:
        node = int(token_nodes[token["id"]])
        trees.append(tree(node, top))
        frontier, cursor = [node], token["base_start"]
        for level in range(top, -1, -1):
            position = cursor
            for item in frontier:
                width = int(counts[item])
                describe(item)
                rows[level].append(
                    {
                        "node": item,
                        "input_token": token["index"],
                        "base_start": position,
                        "base_end": position + width,
                        "carried": level > 0 and item < sizes[level - 1],
                        "introduced": definition.introduced_at(item),
                    }
                )
                position += width
            if level:
                frontier = [
                    child for item in frontier for child in definition.children(item, level)
                ]
        if position != token["base_end"]:
            raise ValueError("hierarchy expansion does not match the base tokens")
    summary = definition.level_summary()
    for record in summary:
        level = record["level"]
        record["distinct_nodes_in_text"] = len({item["node"] for item in rows[level]})
        record["input_vocabulary"] = level == top
    return {
        "levels": summary,
        "rows": [rows[level] for level in range(top, -1, -1)],
        "trees": trees,
        "nodes": {str(node): info for node, info in sorted(nodes.items())},
        "residual_parameters": int(definition.residual_widths.sum()),
    }


def _decoder(base):
    def decode(ids):
        return base.decode(list(ids), skip_special_tokens=False)

    return decode


def render_text_tree(analysis: dict) -> str:
    """A terminal view: segmentation, then each token's composition tree."""
    lines = [
        f"{analysis['name'] or 'Tokenization'}: {len(analysis['input_tokens'])} input "
        f"tokens, {len(analysis['base_tokens'])} base tokens, "
        f"{analysis['byte_count']} bytes"
    ]
    if analysis["levels"]:
        lines.append(
            "Levels: "
            + " -> ".join(
                f"{record['name']}" for record in analysis["levels"]
            )
        )
    nodes = analysis["nodes"]

    def walk(item, depth):
        info = nodes.get(str(item["node"]), {})
        carried = (
            f" carried through L{','.join(map(str, item['carried_through']))}"
            if item["carried_through"]
            else ""
        )
        residual = info.get("residual_width", 0)
        lines.append(
            f"{'  ' * depth}- node {item['node']} {item['text']!r} "
            f"[L{item['level']}{carried}; residual {residual} at "
            f"L{info.get('residual_level', 1)}]"
        )
        for child in item["children"]:
            walk(child, depth + 1)

    lines.append("Segmentation (input tokens):")
    for token, tree in zip(
        analysis["input_tokens"],
        analysis["trees"] or [None] * len(analysis["input_tokens"]),
    ):
        lines.append(f"  #{token['index']} id {token['id']} {token['text']!r}")
        if tree is not None:
            walk(tree, 2)
    return "\n".join(lines)

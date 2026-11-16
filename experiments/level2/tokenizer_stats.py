"""Measure a level-2 tokenizer on the problems the vault names.

Usage: tokenizer_stats.py TOKENIZER.json EVENTS.jsonl [DETAILS.json] [DATA_DIR]

Appends one line of metrics (step 0) to EVENTS.jsonl. The held-out and training text are
the raw caches a training run with the 8k base tokenizer leaves under DATA_DIR (default
data/experiments/poc-fineweb): run it after one.

On held-out FineWeb-Edu (the validation prefix, first ~1M base tokens, BOS/EOS
included as in training):
  compression       base tokens and UTF-8 bytes per token (global positions per base token)
  multi-word        share of vocabulary entries whose text holds a space or newline after
                    its first character (as token_lengths.py), and share of running-text
                    tokens / base tokens inside a token spanning two or more base pre-tokens
  unforced          share of token boundaries (not forced by a special or a line break) where
                    the token plus the next base token is itself a vocabulary entry: the
                    boundary was not decided by the next base token alone. `allowed` also
                    requires that the segmentation could have taken that entry there
                    (word-bounded segmentations never extend across a word start)
  lookahead         as segmentation_lookahead.py: share of boundaries that encoding a prefix
                    gets wrong until >= k further base tokens are visible (sample)
On the first 10M base tokens of the 50M training selection:
  usage             share of entries never used / used < 10 times; share of running-text
                    tokens whose entry is used < 100 times (rarely trained embedding rows)
Plain (non-macro) tokenizers report compression, multi-word and usage only.
"""
import glob, json, os, sys, time
import numpy as np
from datasets import Dataset
from fraise.tokenizer.loading import load_tokenizer
from fraise.tokenizer.macro import word_starts_from_word_ids
from fraise.training.validation.measures import base_byte_lengths

path, events = sys.argv[1], sys.argv[2]
details_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
E = sys.argv[4] if len(sys.argv) > 4 else "data/experiments/poc-fineweb"
tok = load_tokenizer(path)
base = load_tokenizer(f"{E}/tokenizers/bpe-8k.json")
from fraise.tokenizer.loading import tokenizer_fingerprint
BASE_FP = tokenizer_fingerprint(base)


def cache(**want):
    """The raw FineWeb-Edu cache whose request.json matches (found by content, not hash)."""
    for request in sorted(glob.glob(f"{E}/raw/fineweb-edu/raw/*/request.json")):
        r = json.load(open(request))
        if all(r.get(k) == v for k, v in want.items()):
            return os.path.join(os.path.dirname(request), "text.jsonl")
    raise SystemExit(f"no raw cache with {want} under {E}: run training first")
is_macro = hasattr(tok, "decompositions")
V = tok.get_vocab_size()
t0 = time.time()


def texts_until(file, base_budget):
    ds = Dataset.from_json(file)
    out, n = [], 0
    for batch in ds.select_columns("text").iter(batch_size=1000):
        texts = [t for t in batch["text"] if t]
        for t, e in zip(texts, base.encode_batch(texts)):
            out.append(t); n += len(e.ids)
            if n >= base_budget:
                return out
    return out


valid = texts_until(cache(split="validation", bytes=50_000_000), 1_000_000)
train = texts_until(cache(split="train", max_train_tokens=50_000_000, tokenizer_fingerprint=BASE_FP), 10_000_000)
base_bytes = base_byte_lengths(base)

# Vocabulary-level multi-word share (same definition as token_lengths.py).
entry_texts = [tok.decode([i], skip_special_tokens=True) for i in range(V)]
entry_texts = [t for t in entry_texts if t]
vocab_multiword = sum(any(c in t[1:] for c in " \n") for t in entry_texts) / len(entry_texts)

# Held-out text.
n_base = n_tok = n_bytes = 0
multi_tok = multi_base = 0
boundaries = unforced = unforced_allowed = 0
special = np.zeros(base.get_vocab_size(), dtype=bool)
if is_macro:
    entries = {tok.decomposition(i): i for i in range(V)}
    special[[i for i in range(base.get_vocab_size()) if tok._special[i]]] = True
    line_break = tok._line_break
for start in range(0, len(valid), 500):
    chunk = valid[start:start + 500]
    b_encs = base.encode_batch(chunk)
    t_encs = tok.encode_batch(chunk)
    for be, te in zip(b_encs, t_encs):
        ids = be.ids
        starts = word_starts_from_word_ids(be.word_ids)
        n_base += len(ids); n_tok += len(te.ids)
        n_bytes += int(base_bytes[ids].sum())
        if not is_macro:
            continue
        widths = tok.unit_lengths[te.ids]
        ends = np.cumsum(widths); begins = ends - widths
        for k, (b0, w) in enumerate(zip(begins.tolist(), widths.tolist())):
            if w > 1 and starts[b0 + 1:b0 + w].any():
                multi_tok += 1; multi_base += w
            e = b0 + w
            if e >= len(ids):
                continue
            last, nxt = ids[e - 1], ids[e]
            if special[last] or special[nxt] or line_break[last]:
                continue
            boundaries += 1
            if tok.decomposition(te.ids[k]) + (nxt,) in entries:
                unforced += 1
                if not (tok.word_bounded and starts[e]):
                    unforced_allowed += 1

metrics = {
    "step": 0,
    "tok/base_per_token": n_base / n_tok,
    "tok/bytes_per_token": n_bytes / n_tok,
    "tok/positions_per_base_token": n_tok / n_base,
    "tok/vocab_share_multiword": vocab_multiword,
    "tok/vocab_size": V,
}
if is_macro:
    metrics.update({
        "tok/text_share_multiword_tokens": multi_tok / n_tok,
        "tok/text_share_base_in_multiword": multi_base / n_base,
        "tok/unforced_boundaries": unforced / boundaries,
        "tok/unforced_boundaries_allowed": unforced_allowed / boundaries,
        "tok/max_token_length": int(tok.max_macro_length),
    })

# Lookahead by prefix re-encoding, as segmentation_lookahead.py (same sample).
if is_macro:
    boundary = set(tok.boundary_ids)
    hist = np.zeros(26, dtype=np.int64); total = 0
    for text in valid[:60]:
        be = base.encode(text, add_special_tokens=False)
        ids, starts_all = be.ids, word_starts_from_word_ids(be.word_ids)
        segs, cur = [], []
        for i, b in enumerate(ids):
            cur.append(i)
            if b in boundary:
                segs.append(cur); cur = []
        if cur: segs.append(cur)
        for seg in segs[:40]:
            seg = seg[:160]
            s = [ids[i] for i in seg]; ws = starts_all[seg].copy(); ws[0] = True
            full = tok.encode_base_ids(s, ws)
            ends = np.cumsum(tok.unit_lengths[full])
            pre = [None] + [tok.encode_base_ids(s[:L], ws[:L]) for L in range(1, len(s) + 1)]
            for i, e in enumerate(ends):
                e = int(e)
                if e == len(s):
                    continue
                need = None
                for k in range(0, len(s) - e + 1):
                    if all(pre[L][: i + 1] == full[: i + 1] for L in range(e + k, len(s) + 1)):
                        need = k; break
                total += 1; hist[min(need, 25)] += 1
    share = hist / max(total, 1)
    metrics.update({
        "tok/lookahead_sample_boundaries": int(total),
        "tok/lookahead_ge1": float(share[1:].sum()),
        "tok/lookahead_ge2": float(share[2:].sum()),
        "tok/lookahead_ge3": float(share[3:].sum()),
    })

# Usage on 10M training base tokens.
counts = np.zeros(V, dtype=np.int64)
n_train_tok = 0
for start in range(0, len(train), 2000):
    for te in tok.encode_batch(train[start:start + 2000]):
        np.add.at(counts, te.ids, 1); n_train_tok += len(te.ids)
specials = {tok.token_to_id(t) for t in ("<|pad|>", "<|bos|>", "<|eos|>")}
real = np.array([i not in specials for i in range(V)])
metrics.update({
    "tok/vocab_share_unused_10m": float((counts[real] == 0).mean()),
    "tok/vocab_share_under10_10m": float((counts[real] < 10).mean()),
    "tok/text_share_rare_entries_10m": float(counts[counts < 100].sum() / n_train_tok),
    "tok/train_tokens_per_10m_base": n_train_tok,
})
metrics["tok/seconds"] = time.time() - t0
print(json.dumps(metrics, indent=1))
with open(events, "a") as f:
    f.write(json.dumps(metrics) + "\n")
if details_path:
    order = np.argsort(-counts)
    def show(i): return tok.decode([int(i)], skip_special_tokens=False)
    multi = [int(i) for i in order if any(c in show(i)[1:] for c in " \n")][:40]
    det = {"tokenizer": path, "metrics": metrics,
           "most_used": [[show(i), int(counts[i])] for i in order[:40]],
           "most_used_multiword": [[show(i), int(counts[i])] for i in multi],
           "longest_used": [[show(i), int(counts[i])] for i in sorted(np.flatnonzero(counts >= 50), key=lambda i: -len(show(i).encode()))[:30]]}
    json.dump(det, open(details_path, "w"), indent=1, ensure_ascii=False)

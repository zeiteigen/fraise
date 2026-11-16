#!/usr/bin/env bash
# One run of the level-2 (64k) tokenizer study on a fresh GPU runtime (Google Colab, through
# an AblationKit exported notebook), from a clean checkout of this repository.
#
#   bash experiments/level2/colab.sh ARCH-TOKENIZER
#
# ARCH:      hier (Hierarchical-only 64k, both fixes) or fraise (Fraise 64/8, sweep settings)
# TOKENIZER: current | greedy-cross | wordbpe   (committed under tokenizers/)
#            curriculum40k | phrases40k | curriculum | greedy-words
#                                               (trained here, on the same 0.3 GB of FineWeb-Edu)
# REFS (environment, optional): committed tokenizers to measure as well, e.g. "hierarchical-64k",
# logged under ref-<name>/ in the same run.
#
# Writes experiments/level2/out/: events.jsonl (training metrics, then the tokenizer's tok/*
# measurements), training.log, model_meta.json, tokenizer.json and tokstats.json.
set -euo pipefail
VARIANT=$1
ARCH=${VARIANT%%-*}
TOK=${VARIANT#*-}
HERE=experiments/level2
OUT=$HERE/out
D=data/experiments/poc-fineweb
mkdir -p "$OUT" "$D/tokenizers"
export UV_LINK_MODE=copy HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=true

echo "== environment $(date -u +%T)"
command -v uv >/dev/null || python3 -m pip install -q uv
uv sync --frozen --quiet
uv run --frozen python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device: choose a GPU runtime"
name = torch.cuda.get_device_name()
assert torch.cuda.is_bf16_supported(), f"{name} has no BF16; the study trains in BF16 (L4, A100 or H100)"
print("device", name, "torch", torch.__version__)
PY
cp "$HERE/tokenizers/bpe-8k.json" "$D/tokenizers/bpe-8k.json"

echo "== tokenizer $TOK $(date -u +%T)"
make_tokenizer() {
  uv run --frozen create-tokenizer --data-dir "$D" --vocab-size 65536 --sources fineweb-edu \
    --source-size-gb 0.3 --base-tokenizer "$D/tokenizers/bpe-8k.json" "$@" | tail -1
}
case $TOK in
  current) TOKFILE=$HERE/tokenizers/hierarchical-64k.json ;;
  wordbpe) TOKFILE=$HERE/tokenizers/wordbpe-64k.json ;;
  greedy-cross) TOKFILE=$HERE/tokenizers/greedy-cross-64k.json ;;
  greedy-words) TOKFILE=$(make_tokenizer --segmentation greedy --word-bounded) ;;
  curriculum) TOKFILE=$(make_tokenizer --word-bounded-vocab-size 58982) ;;
  curriculum40k) TOKFILE=$(make_tokenizer --word-bounded-vocab-size 40000) ;;
  phrases40k) TOKFILE=$(make_tokenizer --word-bounded-vocab-size 40000 --whole-word-phrases) ;;
  *) echo "unknown tokenizer $TOK" >&2; exit 2 ;;
esac
cp "$TOKFILE" "$OUT/tokenizer.json"
sha256sum "$OUT/tokenizer.json"

case $ARCH in
  hier) MODEL=$HERE/configs/model-64k-finalnorm.json; HP=$HERE/configs/hier-64k-lr3e-3-emb002.json ;;
  fraise) MODEL=$HERE/configs/model-fraise-38m.json; HP=$HERE/configs/fraise-lr3e-3.json ;;
  *) echo "unknown architecture $ARCH" >&2; exit 2 ;;
esac

echo "== train $VARIANT $(date -u +%T)"
uv run --frozen train --data-dir "$D" --save-path "$OUT/runs" --training-hyperparameters "$HP" \
  --model-config "$MODEL" --tokenizer-file "$TOKFILE" --model-name "$VARIANT" --device cuda \
  --events-file "$OUT/events.jsonl"
cp "$OUT/runs/$VARIANT/training.log" "$OUT/runs/$VARIANT/model_meta.json" "$OUT/"

echo "== measure tokenizer $(date -u +%T)"
uv run --frozen python "$HERE/tokenizer_stats.py" "$OUT/tokenizer.json" "$OUT/events.jsonl" \
  --details "$OUT/tokstats.json" --data-dir "$D"
for ref in ${REFS:-}; do
  uv run --frozen python "$HERE/tokenizer_stats.py" "$HERE/tokenizers/$ref.json" "$OUT/events.jsonl" \
    --details "$OUT/tokstats-$ref.json" --data-dir "$D" --prefix "ref-$ref/"
done
echo "== done $(date -u +%T)"

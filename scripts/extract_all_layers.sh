#!/usr/bin/env bash

set -euo pipefail

RUN=""
DATASETS=()
SITE_KIND="resid_post"     # resid_post | resid_pre | attn_out | mlp_out | attn_z
TOKENS="patches"
N_LAYERS=""
SPLIT="train"
BATCH=64
OUT_ROOT="acts"
DTYPE="float16"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --run)        RUN="$2"; shift 2;;
    --datasets)   shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do DATASETS+=("$1"); shift; done;;
    --site-kind)  SITE_KIND="$2"; shift 2;;
    --tokens)     TOKENS="$2"; shift 2;;
    --n-layers)   N_LAYERS="$2"; shift 2;;
    --split)      SPLIT="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --out-root)   OUT_ROOT="$2"; shift 2;;
    --dtype)      DTYPE="$2"; shift 2;;
    *)            EXTRA+=("$1"); shift;;
  esac
done

[ -z "$RUN" ] && { echo "error: --run <checkpoint dir> is required"; exit 1; }
[ ${#DATASETS[@]} -eq 0 ] && { echo "error: --datasets <name...> is required"; exit 1; }

if [ -z "$N_LAYERS" ]; then
  N_LAYERS=$(uv run python -c "import vitlab,sys; print(vitlab.load_model('$RUN').spec.n_layers)")
fi

echo "== model checkpoint: $RUN"
echo "== datasets:         ${DATASETS[*]}  (combined, labels ignored)"
echo "== site kind:        $SITE_KIND   tokens: $TOKENS"
echo "== layers:           0..$((N_LAYERS-1))  (kept separate)"
echo

for (( L=0; L<N_LAYERS; L++ )); do
  SITE="blocks.${L}.${SITE_KIND}"
  echo "--- layer $L : $SITE"
  uv run python scripts/extract_activations.py \
      --run "$RUN" \
      --datasets "${DATASETS[@]}" \
      --site "$SITE" \
      --tokens "$TOKENS" \
      --split "$SPLIT" \
      --batch-size "$BATCH" \
      --dtype "$DTYPE" \
      --out-root "$OUT_ROOT" \
      "${EXTRA[@]}" \
    || { echo "!! layer $L failed -- continuing"; continue; }
done

echo
echo "=== done. one directory per layer under $OUT_ROOT/<model>/<datasets>/"
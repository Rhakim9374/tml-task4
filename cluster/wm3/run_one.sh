#!/usr/bin/env bash
# One WM_3 FNNS variant = one condor process (one GPU). Args: MODE STEPS LAM WP RAIL TAG
set -euo pipefail
REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export REPO
MODE="$1"; STEPS="$2"; LAM="$3"; WP="$4"; RAIL="$5"; TAG="$6"

echo ">> deps"; pip install --quiet opencv-python-headless pillow numpy 2>&1 | tail -1 || true
bash "$REPO/cluster/wm3/fetch_models.sh"

[ -d "$REPO/data/Dataset/watermarked_sources/WM_3" ] || { echo "ERROR: data missing (run cluster/fetch_data.sh)"; exit 1; }

OUT="$REPO/artifacts/wm3_out/$TAG"
echo ">> forging variant TAG=$TAG mode=$MODE steps=$STEPS lam=$LAM wp=$WP rail=$RAIL -> $OUT"
python3 "$REPO/cluster/wm3/fnns_forge.py" \
  --mode "$MODE" --steps "$STEPS" --lam "$LAM" --wp "$WP" --rail "$RAIL" --outdir "$OUT"
echo ">> DONE $TAG. Winner = highest scorecard.json P_to_msrc ABOVE R_floor (held-out transfer)."

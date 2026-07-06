#!/usr/bin/env bash
# WM_5 crack lottery — the last two untested scheme candidates (SepMark, CIN).
#
# LOW prior: WM_5's watermark has anti-correlated RGB channels (a chrominance
# additive scheme), whereas every public deep watermark tested is the wrong tree —
# HiDDeN (0.925 cross-talk), ARWGAN (0.84), PIMoG (0.616), FIN (grayscale). This
# job closes the question: if SepMark or CIN decodes the 25 WM_5 sources at >=0.95
# balanced agreement AND reproduces WM_5's anti-correlated color, WM_5 is crackable
# (re-embed -> ~0.92). Otherwise WM_5 stays on the (already good) additive transplant.
#
# Runs in ~15 min on CPU or one GPU (downloads dominate). Use an interactive job.
#   bash cluster/crack_lottery.sh /path/to/tml-task4        # repo root (has data/Dataset)
set -euo pipefail

REPO="${1:-$PWD}"
WORK="${2:-$REPO/artifacts/crack_lottery}"
DATA="$REPO/data/Dataset"
[ -d "$DATA/watermarked_sources/WM_5" ] || { echo "ERROR: $DATA/watermarked_sources/WM_5 not found"; exit 1; }
mkdir -p "$WORK"; cd "$WORK"

echo ">> installing deps"
python3 -m pip install -q gdown 'numpy<2' pillow torch torchvision kornia pyyaml 2>&1 | tail -2 || true

echo ">> cloning candidate repos"
[ -d SepMark ] || git clone --depth 1 https://github.com/sh1newu/SepMark.git
[ -d CIN ]     || git clone --depth 1 https://github.com/rmpku/CIN.git

echo ">> downloading checkpoints (SepMark ~502MB, CIN ~138MB)"
[ -f SepMark/EC_99.pth ] || gdown 1VGeBtSpxB6zQahZ5ilMl0uaTOz_jORsJ -O SepMark/EC_99.pth
[ -f CIN/cin.pth ]       || gdown 1wqnqhPv92mHwkEI4nMh-sI5aDgh-usr7 -O CIN/cin.pth

echo; echo "======================= SepMark (128/30, conf 50) ======================="
( cd SepMark && python3 "$REPO/scripts/crack_screen.py" sepmark "$DATA" EC_99.pth ) || echo "!! SepMark screen failed"

echo; echo "========================== CIN (128/30, conf 45) ========================="
( cd CIN && python3 "$REPO/scripts/crack_screen.py" cin "$DATA" cin.pth ) || echo "!! CIN screen failed"

echo; echo ">> done. A line printing 'CRACK CANDIDATE' means re-embed & leaderboard-test."
echo ">> No CRACK CANDIDATE => WM_5 is not a public deep scheme; keep the transplant."

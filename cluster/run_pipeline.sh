#!/usr/bin/env bash
# Invoked by cluster/pipeline.sub inside the pytorch docker image. Installs deps,
# extracts each group's watermark, and builds a sweep of submission zips (one per
# LPIPS budget) so a single job yields several candidates to submit hourly.
#
# Extra args are forwarded to scripts.build_submission, e.g.:
#   ... --budgets 0.02,0.03,0.04,0.06
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

pip install --quiet -r requirements.txt
# invisible-watermark drags in the non-headless opencv-python (needs libGL.so.1,
# which the container lacks) and shadows opencv-python-headless. Force headless so
# `import cv2` works without a display library.
pip uninstall -y --quiet opencv-python 2>/dev/null || true
pip install --quiet --force-reinstall --no-deps "opencv-python-headless<5"

# Data must already be at data/Dataset (fetch_data.sh or scp). Fail early if not.
if [ ! -d data/Dataset/clean_targets ]; then
    echo "ERROR: data/Dataset/clean_targets missing. Run cluster/fetch_data.sh first." >&2
    exit 1
fi

mkdir -p artifacts submissions
python -m scripts.extract_watermarks --dataset data/Dataset --out artifacts/wm.npz
python -m scripts.build_submission --dataset data/Dataset --watermarks artifacts/wm.npz \
    --out submissions/sweep.zip "$@"

echo "PIPELINE OK -> submissions/"

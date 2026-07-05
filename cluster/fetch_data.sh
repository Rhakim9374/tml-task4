#!/usr/bin/env bash
# Get the forgery dataset into data/Dataset/ (clean_targets/ + watermarked_sources/).
# Idempotent: does nothing if already unpacked. Run once after cloning.
#
# The data lives in the GATED HuggingFace dataset SprintML/tml2026_task4 as a single
# file, Dataset.zip. Gated => it cannot be wget'd anonymously, and scp'ing it up can
# be killed on the login node. Preferred path: download it on the cluster with your
# HF token (outbound HTTPS, which the login node allows). One-time prerequisites:
#   1. Log into HuggingFace, open the dataset page, and accept/request access (gated).
#   2. Create a READ token at https://huggingface.co/settings/tokens.
# Then:
#   HF_REPO=SprintML/tml2026_task4 HF_TOKEN=hf_xxx bash cluster/fetch_data.sh
#
# Fallbacks: an already-present data/Dataset.zip, or a direct DATA_URL to wget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

mkdir -p data runlogs submissions
DST="data/Dataset"
ZIP="data/Dataset.zip"

if [ -d "$DST/clean_targets" ]; then
    echo "already extracted: $DST"
    exit 0
fi

# 1. Obtain data/Dataset.zip if we don't already have it.
if [ ! -s "$ZIP" ]; then
    if [ -n "${HF_REPO:-}" ]; then
        echo "==> downloading Dataset.zip from HuggingFace ($HF_REPO)"
        pip install --quiet -U huggingface_hub
        # Python API: the huggingface-cli entrypoint is often off PATH. Token is
        # read from HF_TOKEN in the environment.
        HF_REPO="$HF_REPO" python - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HF_REPO"], "Dataset.zip", repo_type="dataset",
                    local_dir="data", token=os.environ.get("HF_TOKEN"))
print("downloaded:", p)
PY
    elif [ -n "${DATA_URL:-}" ]; then
        echo "==> downloading Dataset.zip from DATA_URL"
        wget -q --tries=5 --continue "$DATA_URL" -O "$ZIP"
    else
        echo "ERROR: no data source. Set HF_REPO(+HF_TOKEN), or place $ZIP, or set DATA_URL." >&2
        exit 1
    fi
fi

# 2. Extract. The zip holds clean_targets/ and watermarked_sources/ at the top level.
# Use Python's zipfile (the pytorch container has no `unzip` binary).
echo "==> extracting $ZIP -> $DST"
mkdir -p "$DST"
python - "$ZIP" "$DST" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
n="$(ls "$DST/clean_targets" 2>/dev/null | wc -l | tr -d ' ')"
[ "$n" = "200" ] || { echo "ERROR: expected 200 clean targets, found $n" >&2; exit 1; }
echo "FETCH OK -> $DST ($n clean targets, $(ls -d "$DST"/watermarked_sources/WM_* | wc -l | tr -d ' ') WM groups)"

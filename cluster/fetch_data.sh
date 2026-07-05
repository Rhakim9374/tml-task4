#!/usr/bin/env bash
# Get the forgery dataset into data/Dataset/ (clean_targets/ + watermarked_sources/).
# Idempotent: does nothing if already unpacked. Run once after cloning.
#
# The dataset is GATED on HuggingFace, so it cannot be wget'd anonymously and
# copying it up with scp can be blocked/killed on the login node. Preferred path:
# download it directly on the cluster with your HF token (outbound HTTPS, which the
# login node allows). First accept the dataset's terms on its HF page while logged
# in, then create a read token at https://huggingface.co/settings/tokens.
#
#   HF_REPO=SprintML/tml26_task4 HF_TOKEN=hf_xxx bash cluster/fetch_data.sh
#
# Fallbacks: a local data/Dataset.zip (extracted), or a direct DATA_URL to wget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

mkdir -p data runlogs submissions
DST="data/Dataset"

if [ -d "$DST/clean_targets" ]; then
    echo "already extracted: $DST"
    exit 0
fi

HF_REPO="${HF_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"
DATA_URL="${DATA_URL:-}"
stage="$(mktemp -d)"

if [ -n "$HF_REPO" ]; then
    echo "==> downloading $HF_REPO from HuggingFace (dataset)"
    pip install --quiet -U huggingface_hub
    # Use the Python API (the huggingface-cli entrypoint is often not on PATH,
    # and hf_hub 1.x renamed it to `hf`). Token is read from HF_TOKEN in the env.
    HF_REPO="$HF_REPO" STAGE="$stage" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["HF_REPO"], repo_type="dataset",
                  local_dir=os.environ["STAGE"], token=os.environ.get("HF_TOKEN"))
PY
elif [ -s "data/Dataset.zip" ]; then
    echo "==> extracting local data/Dataset.zip"
    unzip -q -o data/Dataset.zip -d "$stage"
elif [ -n "$DATA_URL" ]; then
    echo "==> downloading DATA_URL"
    wget -q --tries=5 --continue "$DATA_URL" -O "$stage/Dataset.zip"
    unzip -q -o "$stage/Dataset.zip" -d "$stage"
else
    echo "ERROR: no data source. Set HF_REPO(+HF_TOKEN), or place data/Dataset.zip, or set DATA_URL." >&2
    exit 1
fi

# The download may contain a Dataset.zip and/or the folders directly — normalise.
if [ -z "$(find "$stage" -maxdepth 3 -type d -name clean_targets | head -n1)" ]; then
    zip_in="$(find "$stage" -maxdepth 3 -type f -name '*.zip' | head -n1)"
    [ -n "$zip_in" ] && unzip -q -o "$zip_in" -d "$stage"
fi
inner="$(find "$stage" -maxdepth 4 -type d -name clean_targets | head -n1)"
if [ -z "$inner" ]; then
    echo "ERROR: could not find clean_targets/ in the downloaded data" >&2
    exit 1
fi
mkdir -p "$DST"
mv "$(dirname "$inner")"/clean_targets "$(dirname "$inner")"/watermarked_sources "$DST"/
rm -rf "$stage"
echo "FETCH OK -> $DST  ($(ls "$DST/clean_targets" | wc -l | tr -d ' ') targets)"

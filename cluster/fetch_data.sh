#!/usr/bin/env bash
# Fetch and unpack the forgery dataset into data/Dataset/. Idempotent: skips the
# download if the zip is already present and skips extraction if already unpacked.
# Run once after cloning.
#
# The dataset lives on HuggingFace; set DATA_URL to the resolve/ link the course
# provided (kept out of git so the repo carries no credentials or large blobs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
cd "$CODE_DIR"

mkdir -p data runlogs submissions

ZIP="data/Dataset.zip"
DST="data/Dataset"
DATA_URL="${DATA_URL:-}"

if [ -d "$DST" ] && [ -d "$DST/clean_targets" ]; then
    echo "already extracted: $DST"
    exit 0
fi

if [ ! -s "$ZIP" ]; then
    if [ -z "$DATA_URL" ]; then
        echo "ERROR: $ZIP missing and DATA_URL not set." >&2
        echo "Set DATA_URL to the HuggingFace resolve/ link, e.g.:" >&2
        echo "  DATA_URL=https://huggingface.co/datasets/<...>/Dataset.zip bash cluster/fetch_data.sh" >&2
        exit 1
    fi
    echo "==> downloading dataset zip"
    wget -q --tries=5 --continue "$DATA_URL" -O "$ZIP" \
        || { rm -f "$ZIP"; echo "FAILED: $DATA_URL" >&2; exit 1; }
fi

echo "==> extracting $ZIP -> $DST"
tmp="$(mktemp -d)"
unzip -q -o "$ZIP" -d "$tmp"
# The zip's top-level folder may vary; normalise to data/Dataset/.
inner="$(find "$tmp" -maxdepth 2 -type d -name clean_targets | head -n1)"
if [ -z "$inner" ]; then
    echo "ERROR: could not find clean_targets/ inside the zip" >&2
    exit 1
fi
mkdir -p "$DST"
mv "$(dirname "$inner")"/* "$DST"/
rm -rf "$tmp"
echo "FETCH OK -> $DST"

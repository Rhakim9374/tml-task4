#!/usr/bin/env bash
# Fetch the two decoders the WM_3 FNNS job needs, into $REPO/artifacts/wm3_models/.
# Idempotent: skips anything already present. Runs inside the pytorch docker (has HTTPS).
set -euo pipefail
REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
M="$REPO/artifacts/wm3_models"
mkdir -p "$M"

# RoSteALS full ControlAE checkpoint (~520MB) — contains ae.* + control.* + decoder.* (resnet50).
if [ ! -s "$M/rosteals_control.ckpt" ]; then
  echo ">> downloading RoSteALS ControlAE checkpoint (~520MB)"
  wget -q -O "$M/rosteals_control.ckpt" \
    "https://kahlan.cvssp.org/data/Flickr25K/tubui/cvpr23_wmf/epoch=000017-step=000449999.ckpt"
fi

# PIMoG repo (for model.py -> Encoder_Decoder) + its pretrained ScreenShooting decoder (~5MB).
if [ ! -d "$M/PIMoG" ]; then
  echo ">> cloning PIMoG"
  git clone --depth 1 \
    "https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw" \
    "$M/PIMoG" || echo "clone failed — see README (scp model.py + pimog.pth)"
fi
if [ ! -s "$M/pimog.pth" ]; then
  found="$(find "$M/PIMoG" -name '*mask_99*.pth' 2>/dev/null | head -1 || true)"
  if [ -n "$found" ]; then cp "$found" "$M/pimog.pth"; else
    echo "!! PIMoG pretrained (Encoder_Decoder_Model_mask_99.pth, 5MB) not in the repo."
    echo "!! scp it from your laptop to $M/pimog.pth (see cluster/wm3/README.md)."
  fi
fi
echo ">> models ready in $M"
ls -la "$M" || true

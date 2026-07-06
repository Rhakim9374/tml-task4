#!/usr/bin/env bash
# WM_5 crack lottery — the last two untested scheme candidates (SepMark, CIN).
# Invoked by cluster/crack_lottery.sub inside the pytorch docker image (which has
# outbound HTTPS, like run_pipeline.sh's pip install). Also runnable directly on an
# interactive node:  bash cluster/crack_lottery.sh <repo-root>
#
# LOW prior: WM_5's watermark has anti-correlated RGB channels (a chrominance
# additive scheme). Every public deep watermark tested is the wrong tree —
# HiDDeN (0.925 cross-talk), ARWGAN (0.84), PIMoG (0.616), FIN (grayscale). This
# job closes the question: if SepMark or CIN decodes the 25 WM_5 sources at >=0.95
# balanced agreement AND reproduces WM_5's anti-correlated color, WM_5 is crackable
# (re-embed -> ~0.92). Otherwise WM_5 stays on the (already good) additive transplant.
set -euxo pipefail

REPO="${1:-$PWD}"
DATA="$REPO/data/Dataset"
WORK="$REPO/artifacts/crack_lottery"
[ -d "$DATA/watermarked_sources/WM_5" ] || { echo "ERROR: $DATA/watermarked_sources/WM_5 not found (run cluster/fetch_data.sh)"; exit 1; }
mkdir -p "$WORK"; cd "$WORK"

echo ">> deps"
pip install --quiet gdown kornia pyyaml 2>&1 | tail -1 || true
# guard against a cv2 pulled in by a transitive dep needing libGL (container lacks it)
pip install --quiet --force-reinstall --no-deps "opencv-python-headless<5" 2>/dev/null || true

echo ">> fetching candidate repos (tarball via HTTPS, no git needed)"
python3 - <<'PY'
import urllib.request, tarfile, io, os
def fetch(user, repo, branches=("main","master")):
    if os.path.isdir(repo):
        print("have", repo); return
    for br in branches:
        try:
            url = f"https://codeload.github.com/{user}/{repo}/tar.gz/refs/heads/{br}"
            data = urllib.request.urlopen(url, timeout=120).read()
            tarfile.open(fileobj=io.BytesIO(data)).extractall(".")
            os.rename(f"{repo}-{br}", repo); print("fetched", repo, "@", br); return
        except Exception as e:
            print(" ", repo, br, "->", e)
    raise SystemExit(f"could not fetch {repo}")
fetch("sh1newu", "SepMark")
fetch("rmpku", "CIN")
PY

echo ">> downloading checkpoints (SepMark ~502MB, CIN ~138MB) via gdown Python API"
# The gdown CLI entry point is often off PATH in the container; use the API directly.
python3 - <<'PY'
import os, gdown
jobs = [("1VGeBtSpxB6zQahZ5ilMl0uaTOz_jORsJ", "SepMark/EC_99.pth"),
        ("1wqnqhPv92mHwkEI4nMh-sI5aDgh-usr7", "CIN/cin.pth")]
for fid, out in jobs:
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print("have", out); continue
    gdown.download(id=fid, output=out, quiet=False, fuzzy=True)
PY

set +e
echo; echo "===================== SepMark (128/30, conf 50) ====================="
( cd SepMark && python3 "$REPO/scripts/crack_screen.py" sepmark "$DATA" EC_99.pth )
echo; echo "======================== CIN (128/30, conf 45) ======================"
( cd CIN && python3 "$REPO/scripts/crack_screen.py" cin "$DATA" cin.pth )

echo; echo ">> A line printing 'CRACK CANDIDATE' => re-embed & leaderboard-test."
echo ">> No CRACK CANDIDATE => WM_5 is not a public deep scheme; keep the transplant."

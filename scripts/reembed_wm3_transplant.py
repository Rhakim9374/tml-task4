"""WM_3 forgery = improved additive-transplant (the method that beat every decoder-based approach).

WM_3's watermark could not be matched to any public scheme (see plans/RESULTS_LOG.md), but it has a
recoverable fixed additive component. We estimate the fixed watermark pattern W by the per-pixel MEDIAN
of the NLM denoiser residual over the group's 25 watermarked sources (content cancels across the diverse
sources; the shared watermark survives), then imprint it onto the clean targets:  forged = clip(clean + alpha*W).

alpha=2.0 was selected by a held-out cross-decoder judge (RoSteALS + PIMoG, neither used to build the
forgery): it reproduces the real sources' watermark signature at the knee of the transfer-vs-LPIPS curve.
On the leaderboard this lifted WM_3 from near-failing to strong. Sweep alpha with --alpha to retune.

    python scripts/reembed_wm3_transplant.py --alpha 2.0 --out artifacts/wm3_transplant
"""
import argparse
import glob
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nlm_residual(path):
    img = cv2.imread(path).astype(np.float32)
    den = cv2.fastNlMeansDenoisingColored(cv2.imread(path), None, 10, 10, 7, 21).astype(np.float32)
    return img - den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{REPO}/data/Dataset")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--out", default=f"{REPO}/artifacts/wm3_transplant")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    srcs = sorted(glob.glob(f"{a.dataset}/watermarked_sources/WM_3/*.png"))
    W = np.median(np.stack([nlm_residual(p) for p in srcs]), axis=0)   # fixed-pattern estimate (BGR)
    rmss = []
    for n in range(51, 76):
        clean = cv2.imread(f"{a.dataset}/clean_targets/{n}.png").astype(np.float32)
        forged = np.clip(clean + a.alpha * W, 0, 255)
        cv2.imwrite(f"{a.out}/{n}.png", forged.round().astype(np.uint8))
        rmss.append(float(np.sqrt(((forged - clean) ** 2).mean())))
    print(f"WM_3 transplant alpha={a.alpha}: wrote 25 imgs to {a.out}  mean imprint rms={np.mean(rmss):.2f} gl")


if __name__ == "__main__":
    main()

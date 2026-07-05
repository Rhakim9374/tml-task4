"""Diagnostics that characterise how forgeable each watermark group is.

For every WM group we split its 25 sources into two disjoint halves, estimate a
watermark residual from each half, and measure the normalised cross-correlation
between the two estimates. A high correlation means the watermark is a
consistent, roughly additive pattern (content averages out, watermark remains)
that a residual-transplant attack can reproduce; a near-zero correlation means
the watermark is content-dependent or lives in a transform domain the estimator
does not isolate, and needs a stronger extractor.

    python -m scripts.analyze --dataset data/Dataset

This is a read-only analysis; it writes nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data import iter_groups, load_group_sources


def box_blur(x: np.ndarray, k: int) -> np.ndarray:
    """Reflect-padded k x k box blur over an (H, W, C) array."""
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for dy in range(k):
        for dx in range(k):
            out += xp[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return out / (k * k)


def highpass_residual(group_stack: np.ndarray, k: int) -> np.ndarray:
    """Mean over images of (image - box_blur(image)) — a watermark estimate."""
    return np.mean([im - box_blur(im, k) for im in group_stack], axis=0)


def cross_half_corr(group_stack: np.ndarray, k: int) -> float:
    h1, h2 = group_stack[0::2], group_stack[1::2]
    r1 = highpass_residual(h1, k)
    r2 = highpass_residual(h2, k)
    return float(np.corrcoef(r1.ravel(), r2.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser(description="Watermark forgeability diagnostics")
    ap.add_argument("--dataset", default="data/Dataset", type=Path)
    ap.add_argument("--kernels", default="3,5,9,17")
    args = ap.parse_args()

    kernels = [int(k) for k in args.kernels.split(",")]
    header = "group  res      " + "".join(f"k={k:<2}    " for k in kernels)
    print(header)
    print("-" * len(header))
    for g in iter_groups(args.dataset):
        stack = load_group_sources(g)
        h, w = stack.shape[1:3]
        corrs = [cross_half_corr(stack, k) for k in kernels]
        cells = "".join(f"{c:6.3f}  " for c in corrs)
        print(f"{g.name:6} {h}x{w:<4} {cells}")

    print(
        "\nInterpretation: corr >> 0 (e.g. WM_4) => consistent additive watermark,\n"
        "well suited to residual transplant; corr ~ 0 => needs a stronger extractor."
    )


if __name__ == "__main__":
    main()

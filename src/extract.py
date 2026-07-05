"""Estimate each group's watermark pattern by averaging denoiser residuals.

For a group of N images that share one watermark message, we compute the
per-image residual ``x_i - denoise(x_i)`` and aggregate it across the group.
If the watermark is (approximately) a fixed additive pattern w, then
``x_i = c_i + w`` and ``residual_i ~= w + (content leakage)``; the content
leakage is roughly zero-mean across diverse images, so the aggregate converges
to w. We pick the denoiser per group by a *cross-half consistency* score: split
the group in two, estimate w from each half, and correlate — high correlation
means the estimate is a real, reproducible pattern rather than content noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import denoise


def aggregate(residuals: np.ndarray, method: str = "mean") -> np.ndarray:
    """Combine per-image residuals (N, H, W, 3) into one watermark estimate."""
    if method == "mean":
        return residuals.mean(0)
    if method == "median":
        return np.median(residuals, axis=0)
    if method == "trimmed":
        # drop the most extreme 20% per pixel before averaging (robust to outliers)
        lo, hi = np.percentile(residuals, [10, 90], axis=0)
        clipped = np.clip(residuals, lo, hi)
        return clipped.mean(0)
    raise ValueError(f"unknown aggregate method {method!r}")


def estimate(stack: np.ndarray, denoiser: str, method: str = "mean") -> np.ndarray:
    """Watermark estimate w-hat for a group stack (N, H, W, 3)."""
    resids = np.stack([denoise.residual(im, denoiser) for im in stack])
    return aggregate(resids, method)


def cross_half_consistency(stack: np.ndarray, denoiser: str, method: str = "mean") -> float:
    """Correlation between watermark estimates from two disjoint halves.

    A denoiser-agnostic measure of how forgeable (how additive/reproducible)
    the watermark is. ~0 => the residual is mostly content, not a shared pattern.
    """
    a = estimate(stack[0::2], denoiser, method)
    b = estimate(stack[1::2], denoiser, method)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@dataclass
class Extraction:
    group: str
    denoiser: str
    method: str
    consistency: float
    watermark: np.ndarray  # (H, W, 3) float residual estimate

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.watermark ** 2)))


def best_extraction(
    group_name: str,
    stack: np.ndarray,
    denoisers: list[str] | None = None,
    method: str = "mean",
) -> Extraction:
    """Try each denoiser, keep the one with the highest cross-half consistency."""
    denoisers = denoisers or denoise.available()
    scored = []
    for d in denoisers:
        try:
            c = cross_half_consistency(stack, d, method)
        except Exception as e:
            # Surface failures — a silently skipped denoiser (e.g. nlmeans dying
            # under a bad numpy build) would quietly pick a weaker fallback.
            print(f"  [warn] denoiser {d!r} failed for {group_name}: {type(e).__name__}: {e}")
            continue
        scored.append((c, d))
    if not scored:
        raise RuntimeError(f"no denoiser succeeded for {group_name}")
    consistency, denoiser = max(scored)
    w = estimate(stack, denoiser, method)
    return Extraction(group_name, denoiser, method, consistency, w)

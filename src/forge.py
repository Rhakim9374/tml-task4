"""Imprint an estimated watermark onto clean targets under a quality budget.

We add a scaled copy of the watermark estimate to each clean image:
``forged = clip(clean + alpha * w_hat)``. The scale ``alpha`` is chosen per
image so the perceptual distance hits a target LPIPS budget — the only quality
signal we can measure locally (bit-accuracy needs the hidden decoder). Because
bit-accuracy generally rises with watermark strength while quality falls, the
best operating point is the largest perturbation the quality budget allows; we
sweep that single budget on the leaderboard.

LPIPS is monotonic in ``alpha`` for a fixed direction, so a per-image bisection
finds the scale that meets the budget. ``alpha`` is capped so the injected
perturbation never exceeds ``cap_rms`` gray levels (protects the hard groups,
whose estimate is weak and would otherwise be amplified into visible noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import quality


@dataclass
class ForgeResult:
    forged: np.ndarray  # (H, W, 3) float in [0, 255]
    alpha: float
    lpips: float
    psnr: float


def _apply(clean: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(clean + alpha * delta, 0.0, 255.0)


def forge_to_lpips(
    clean: np.ndarray,
    delta: np.ndarray,
    target_lpips: float,
    net: str = "alex",
    cap_rms: float = 24.0,
    iters: int = 12,
) -> ForgeResult:
    """Scale ``delta`` onto ``clean`` to reach ``target_lpips`` (bisection)."""
    rms = float(np.sqrt(np.mean(delta ** 2))) + 1e-8
    alpha_max = cap_rms / rms

    hi_img = _apply(clean, delta, alpha_max)
    hi_lpips = quality.lpips_distance(clean, hi_img, net)
    if hi_lpips <= target_lpips:
        # even at the cap we stay under budget — use the cap (weak watermark).
        return ForgeResult(hi_img, alpha_max, hi_lpips, quality.psnr(clean, hi_img))

    lo, hi = 0.0, alpha_max
    best = hi_img
    best_a, best_l = alpha_max, hi_lpips
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        img = _apply(clean, delta, mid)
        l = quality.lpips_distance(clean, img, net)
        best, best_a, best_l = img, mid, l
        if l > target_lpips:
            hi = mid
        else:
            lo = mid
    return ForgeResult(best, best_a, best_l, quality.psnr(clean, best))


def forge_to_alpha(clean: np.ndarray, delta: np.ndarray, alpha: float) -> ForgeResult:
    """Apply a fixed scale (no LPIPS needed) — for quick, decoder-free runs."""
    img = _apply(clean, delta, alpha)
    return ForgeResult(img, alpha, float("nan"), quality.psnr(clean, img))

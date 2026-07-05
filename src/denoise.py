"""Denoisers used to isolate the watermark residual.

The forgery attack rests on a steganalysis idea (Yang et al., "Can Simple
Averaging Defeat Modern Watermarks?", NeurIPS 2024): for a content-agnostic
watermark, ``image - denoise(image)`` keeps the watermark and discards most of
the image content, so averaging that residual over many images that share one
message reveals the watermark pattern. The denoiser choice is what matters —
a self-similarity denoiser (non-local means) or a wavelet shrinkage denoiser
removes content far better than a plain box/median filter, which is why they
expose watermarks the naive filter misses (see ``scripts/analyze.py``).

All functions here take and return float arrays in [0, 255], shape (H, W, 3),
and return the *denoised* image; the residual is ``image - denoise(image)``.
Deep / learned denoisers (GPU) are added in ``src/denoise_deep.py`` and share
this interface via ``register``.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

Denoiser = Callable[[np.ndarray], np.ndarray]
_REGISTRY: dict[str, Denoiser] = {}


def register(name: str):
    def deco(fn: Denoiser) -> Denoiser:
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Denoiser:
    if name not in _REGISTRY:
        raise KeyError(f"unknown denoiser {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


@register("median")
def _median(im: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter

    return median_filter(im, size=(3, 3, 1))


@register("gaussian")
def _gaussian(im: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(im, sigma=(0.8, 0.8, 0))


@register("tv")
def _tv(im: np.ndarray) -> np.ndarray:
    from skimage.restoration import denoise_tv_chambolle

    return denoise_tv_chambolle(im / 255.0, weight=0.08, channel_axis=-1) * 255.0


@register("wavelet")
def _wavelet(im: np.ndarray) -> np.ndarray:
    from skimage.restoration import denoise_wavelet

    return denoise_wavelet(im / 255.0, channel_axis=-1, rescale_sigma=True) * 255.0


@register("nlmeans")
def _nlmeans(im: np.ndarray) -> np.ndarray:
    from skimage.restoration import denoise_nl_means, estimate_sigma

    x = im / 255.0
    sigma = float(np.mean(estimate_sigma(x, channel_axis=-1)))
    den = denoise_nl_means(
        x, h=1.15 * sigma, sigma=sigma, fast_mode=True,
        patch_size=5, patch_distance=6, channel_axis=-1,
    )
    return den * 255.0


@register("bilateral")
def _bilateral(im: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.bilateralFilter(im.astype(np.float32), d=7, sigmaColor=40, sigmaSpace=7)


def residual(im: np.ndarray, denoiser: str) -> np.ndarray:
    """image - denoise(image); the per-image watermark estimate."""
    return im - get(denoiser)(im)

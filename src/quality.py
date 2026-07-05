"""Visual-quality metrics used by the leaderboard's quality score.

The server scores each forged image with

    S_qlt = exp(-8 * LPIPS(clean, forged))

so LPIPS is the only quality signal we can reproduce locally (bit-accuracy needs
the hidden decoder). We use the reference ``lpips`` package (AlexNet backbone,
the common default) as a proxy; the server's exact backbone is unknown, so treat
absolute values as indicative and rely on it mainly for *relative* comparisons
and for scaling perturbations to a fixed quality budget.
"""

from __future__ import annotations

import numpy as np

_lpips_model = None


def _get_lpips(net: str = "alex"):
    global _lpips_model
    if _lpips_model is None:
        import lpips  # imported lazily so analysis without torch still works
        import torch

        _lpips_model = lpips.LPIPS(net=net)
        _lpips_model.eval()
        for p in _lpips_model.parameters():
            p.requires_grad_(False)
    return _lpips_model


def _to_tensor(arr: np.ndarray):
    """(H, W, 3) uint8-range float -> (1, 3, H, W) tensor in [-1, 1]."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).unsqueeze(0)


def lpips_distance(clean: np.ndarray, forged: np.ndarray, net: str = "alex") -> float:
    """LPIPS perceptual distance between two [0, 255] RGB arrays."""
    import torch

    model = _get_lpips(net)
    with torch.no_grad():
        d = model(_to_tensor(clean), _to_tensor(forged))
    return float(d.reshape(-1)[0])


def quality_score(lpips_value: float) -> float:
    """Leaderboard quality term S_qlt = exp(-8 * LPIPS)."""
    return float(np.exp(-8.0 * lpips_value))


def detection_score(bit_accuracy: float) -> float:
    """Leaderboard strength term S_det = max(0, (BitAcc - 0.5) * 2)."""
    return max(0.0, (bit_accuracy - 0.5) * 2.0)


def psnr(clean: np.ndarray, forged: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) — a cheap, decoder-free sanity metric."""
    mse = np.mean((clean.astype(np.float64) - forged.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))

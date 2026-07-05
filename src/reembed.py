"""Scheme-identification forgery: decode a group's shared message with the actual
public watermark library, then re-embed that message into the clean targets with
the same library's encoder.

When a group's watermark is a known public scheme, this is far stronger than the
blind additive transplant: the evaluator's own decoder reads our re-embedded
message near-perfectly. Empirically **WM_2 is RivaGAN** (invisible-watermark):
its 25 sources decode to one balanced 32-bit message with 0.99 agreement, and a
re-embed round-trip gives bit-accuracy 1.0 at LPIPS ~0.01-0.02.

Identifying a scheme is validated three ways to avoid the degenerate false
positives raw decoders produce: (1) high cross-image bit agreement, (2) a
*balanced* message (not all-0/all-1), (3) a re-embed round-trip that recovers it.
"""

from __future__ import annotations

import numpy as np

# group -> scheme name; extend as more groups are identified.
GROUP_SCHEME = {
    "WM_2": "rivaGan",
}

_RIVA_LOADED = False


def _ensure_riva():
    global _RIVA_LOADED
    if not _RIVA_LOADED:
        from imwatermark import WatermarkDecoder, WatermarkEncoder

        WatermarkEncoder.loadModel()
        WatermarkDecoder.loadModel()
        _RIVA_LOADED = True


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def extract_message(source_paths, scheme: str, length: int = 32) -> np.ndarray:
    """Majority-vote the shared message across a group's watermarked sources."""
    from imwatermark import WatermarkDecoder

    from src.data import load_rgb

    if scheme != "rivaGan":
        raise NotImplementedError(f"scheme {scheme!r} not wired up yet")
    _ensure_riva()
    rows = []
    for p in source_paths:
        bgr = _to_bgr(load_rgb(p))
        if min(bgr.shape[:2]) < 256:  # RivaGAN needs >=256px
            continue
        rows.append(list(WatermarkDecoder("bits", length).decode(bgr, "rivaGan")))
    rows = np.array(rows, dtype=np.int32)
    return (rows.mean(0) >= 0.5).astype(np.int32)


def message_agreement(source_paths, scheme: str, length: int = 32) -> tuple[float, float]:
    """(cross-image agreement, message balance) — for validating an identification."""
    from imwatermark import WatermarkDecoder

    from src.data import load_rgb

    _ensure_riva()
    rows = []
    for p in source_paths:
        bgr = _to_bgr(load_rgb(p))
        if min(bgr.shape[:2]) < 256:
            continue
        rows.append(list(WatermarkDecoder("bits", length).decode(bgr, "rivaGan")))
    rows = np.array(rows, dtype=np.int32)
    maj = (rows.mean(0) >= 0.5).astype(int)
    return float((rows == maj).mean()), float(maj.mean())


def embed_message(clean_rgb: np.ndarray, message: np.ndarray, scheme: str) -> np.ndarray:
    """Return the clean image (float RGB [0,255]) with the message embedded."""
    from imwatermark import WatermarkEncoder

    if scheme != "rivaGan":
        raise NotImplementedError(f"scheme {scheme!r} not wired up yet")
    _ensure_riva()
    enc = WatermarkEncoder()
    enc.set_watermark("bits", message.tolist())
    wm_bgr = enc.encode(_to_bgr(clean_rgb), "rivaGan")
    return _to_rgb(wm_bgr)

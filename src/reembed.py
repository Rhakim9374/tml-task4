"""Scheme-identification forgery: decode a group's shared message with the actual
public watermark library, then re-embed that message into the clean targets with
the same library's encoder.

When a group's watermark is a known public scheme, this is far stronger than the
blind additive transplant: the evaluator's own decoder reads our re-embedded
message directly. Identified so far (invisible-watermark library):
  * WM_2 = RivaGAN  — 0.99 cross-image agreement, re-embed round-trip bitacc 1.0,
    LPIPS ~0.01-0.02.
  * WM_1 = DwtDct (32-bit) — 0.83 agreement (vs 0.58 baseline), round-trip bitacc
    ~0.86 (DwtDct's inherent reliability), LPIPS ~0.008.

Every identification is validated against degenerate false positives (raw decoders
happily emit all-0/all-1): we require high cross-image agreement AND a balanced
message (bit-fraction in ~[0.3, 0.7]) AND a re-embed round-trip that recovers it.
"""

from __future__ import annotations

import numpy as np

# group -> (scheme method, message length). Extend as groups are identified.
GROUP_SCHEME = {
    "WM_2": ("rivaGan", 32),
    "WM_1": ("dwtDct", 32),
}

_RIVA_LOADED = False


def _ensure_riva():
    """RivaGAN needs its ONNX model loaded once; DwtDct needs nothing."""
    global _RIVA_LOADED
    if not _RIVA_LOADED:
        from imwatermark import WatermarkDecoder, WatermarkEncoder

        WatermarkEncoder.loadModel()
        WatermarkDecoder.loadModel()
        _RIVA_LOADED = True


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8)[:, :, ::-1])


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32)


def _min_side_ok(bgr: np.ndarray, method: str) -> bool:
    # RivaGAN requires >=256px; DwtDct works down to small sizes.
    return not (method == "rivaGan" and min(bgr.shape[:2]) < 256)


def extract_message(source_paths, scheme: str, length: int) -> np.ndarray:
    """Majority-vote the shared message across a group's watermarked sources."""
    from imwatermark import WatermarkDecoder

    from src.data import load_rgb

    if scheme == "rivaGan":
        _ensure_riva()
    rows = []
    for p in source_paths:
        bgr = _to_bgr(load_rgb(p))
        if not _min_side_ok(bgr, scheme):
            continue
        rows.append(list(WatermarkDecoder("bits", length).decode(bgr, scheme)))
    rows = np.array(rows, dtype=np.int32)
    return (rows.mean(0) >= 0.5).astype(np.int32)


def message_agreement(source_paths, scheme: str, length: int) -> tuple[float, float]:
    """(cross-image agreement, message balance) — for validating an identification."""
    from imwatermark import WatermarkDecoder

    from src.data import load_rgb

    if scheme == "rivaGan":
        _ensure_riva()
    rows = []
    for p in source_paths:
        bgr = _to_bgr(load_rgb(p))
        if not _min_side_ok(bgr, scheme):
            continue
        rows.append(list(WatermarkDecoder("bits", length).decode(bgr, scheme)))
    rows = np.array(rows, dtype=np.int32)
    maj = (rows.mean(0) >= 0.5).astype(int)
    return float((rows == maj).mean()), float(maj.mean())


def embed_message(clean_rgb: np.ndarray, message: np.ndarray, scheme: str) -> np.ndarray:
    """Return the clean image (float RGB [0,255]) with the message embedded."""
    from imwatermark import WatermarkEncoder

    if scheme == "rivaGan":
        _ensure_riva()
    enc = WatermarkEncoder()
    enc.set_watermark("bits", message.tolist())
    wm_bgr = enc.encode(_to_bgr(clean_rgb), scheme)
    return _to_rgb(wm_bgr)

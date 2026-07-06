"""Self-contained honest transfer-prediction metric for the WM_1/WM_3 scheme hunt.

The 25 watermarked sources per group are the EVALUATOR's own watermarked images --
external ground truth. A candidate decoder that reads all 25 as the same balanced
message, and keeps reading it under mild JPEG/resize, is decoding the real watermark.
This is NOT self-referential like an encode->decode round-trip (which is 1.0 for any
matched pair). Calibration on this dataset: a WRONG decoder floors at ~0.58-0.60
agreement; a REAL crack (verified on WM_2 RivaGAN) sits at agree>=0.99 with robustness
staying ~0.96+. Gate a crack on: agree>=0.99 AND 0.2<bal<0.8 AND robustness high.

Each scheme supplies decode(img_bgr_uint8) -> 1-D np.array of bits (0/1). Input is a
cv2 BGR uint8 image; the decoder converts/normalizes/resizes as its scheme requires.
"""
import glob

import cv2
import numpy as np


def augment(img, name):
    if name == "orig":
        return img
    if name.startswith("jpeg"):
        q = int(name[4:])
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if name == "resize90":
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(name)


def _consensus(msgs):
    M = np.stack(msgs)
    c = (M.mean(0) >= 0.5).astype(int)
    return c, float((M == c).mean())


def honest_metrics(paths, decode_fn, augs=("jpeg80", "resize90")):
    base = [cv2.imread(p) for p in paths]
    msgs = [np.asarray(decode_fn(b)).astype(int).reshape(-1) for b in base]
    n = min(len(m) for m in msgs)
    msgs = [m[:n] for m in msgs]  # guard against ragged lengths
    c, agree = _consensus(msgs)
    M = np.stack(msgs)
    out = {"agree": agree, "bal": float(c.mean()), "nbits": int(c.size),
           "subset": float(((M[: len(M) // 2].mean(0) >= 0.5).astype(int) == c).mean())}
    for a in augs:
        ms = [np.asarray(decode_fn(augment(b, a))).astype(int).reshape(-1)[:n] for b in base]
        out[f"rob_{a}"] = float((np.stack(ms) == c).mean())
    return out, c


# Fixed group -> clean-target id mapping (WM_1->1..25, WM_2->26..50, ... WM_8->176..200).
GROUP_TARGETS = {f"WM_{i}": (25 * (i - 1) + 1, 25 * i) for i in range(1, 9)}


def run_group(scheme, data, group, decode_fn, augs=("jpeg80", "resize90")):
    """Decode a group's watermarked sources AND its clean (unwatermarked) targets, and
    report the excess agreement `delta = agree_src - agree_clean`. CNN decoders have a
    high content-driven floor (~0.70), so raw agreement is misleading -- only the excess
    over the SAME decoder's reading of clean images is watermark signal. A real crack:
    agree>=0.99, balanced, robust, AND delta large (the clean images decode near chance)."""
    paths = sorted(glob.glob(f"{data}/watermarked_sources/{group}/*.png"))
    try:
        m, c = honest_metrics(paths, decode_fn, augs=augs)
        a, b = GROUP_TARGETS[group]
        clean_paths = [f"{data}/clean_targets/{n}.png" for n in range(a, b + 1)]
        mc, _ = honest_metrics(clean_paths, decode_fn, augs=())  # content floor for THIS decoder
    except Exception as e:
        print(f"RESULT scheme={scheme} group={group} ERROR {type(e).__name__}: {e}", flush=True)
        return None
    m["agree_clean"] = mc["agree"]
    m["delta"] = m["agree"] - mc["agree"]
    robs = [v for k, v in m.items() if k.startswith("rob_")]
    balanced = 0.2 < m["bal"] < 0.8
    clean = (m["agree"] >= 0.99 and balanced and (min(robs) if robs else 0) >= 0.90 and m["delta"] >= 0.20)
    if clean:
        flag = "  <<< CLEAN CRACK"
    elif not balanced:
        flag = "  (degenerate: bal out of range -> ignore agree)"
    elif m["delta"] >= 0.15:
        flag = f"  ~real partial signal (delta={m['delta']:.2f})"
    elif m["agree"] >= 0.90:
        flag = f"  (high agree but delta={m['delta']:.2f} -> mostly content floor)"
    else:
        flag = ""
    body = " ".join(f"{k}={m[k]:.3f}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in m)
    print(f"RESULT scheme={scheme} group={group} {body}{flag}", flush=True)
    return m

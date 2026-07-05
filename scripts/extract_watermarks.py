"""Estimate every group's watermark pattern and cache it to artifacts/.

    python -m scripts.extract_watermarks --dataset data/Dataset --out artifacts/wm.npz

For each WM group it tries all registered denoisers, keeps the one with the
highest cross-half consistency, and stores the averaged watermark estimate plus
metadata. ``scripts.build_submission`` consumes this cache. Add ``--denoiser
nlmeans`` to force one denoiser, or ``--aggregate median`` for a robust mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import denoise
from src.data import iter_groups, load_group_sources
from src.extract import best_extraction, cross_half_consistency, estimate


def main():
    ap = argparse.ArgumentParser(description="Extract per-group watermark estimates")
    ap.add_argument("--dataset", default="data/Dataset", type=Path)
    ap.add_argument("--out", default="artifacts/wm.npz", type=Path)
    ap.add_argument("--denoiser", default="auto",
                    help="'auto' (best by consistency) or a name from src.denoise")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "median", "trimmed"])
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, np.ndarray] = {}
    meta_lines = []
    print(f"{'group':6} {'denoiser':10} {'consistency':>12} {'rms':>8}")
    print("-" * 40)
    for g in iter_groups(args.dataset):
        stack = load_group_sources(g)
        if args.denoiser == "auto":
            ex = best_extraction(g.name, stack, denoise.available(), method=args.aggregate)
        else:
            w = estimate(stack, args.denoiser, args.aggregate)
            c = cross_half_consistency(stack, args.denoiser, args.aggregate)
            from src.extract import Extraction
            ex = Extraction(g.name, args.denoiser, args.aggregate, c, w)
        store[f"{g.name}"] = ex.watermark.astype(np.float32)
        store[f"{g.name}__meta"] = np.array(
            [ex.denoiser, ex.method, f"{ex.consistency:.4f}", f"{ex.rms:.4f}"])
        print(f"{g.name:6} {ex.denoiser:10} {ex.consistency:12.3f} {ex.rms:8.3f}")
        meta_lines.append(f"{g.name}: denoiser={ex.denoiser} consistency={ex.consistency:.3f}")

    np.savez_compressed(args.out, **store)
    print(f"\nsaved {len(iter_groups(args.dataset))} watermark estimates -> {args.out}")


if __name__ == "__main__":
    main()

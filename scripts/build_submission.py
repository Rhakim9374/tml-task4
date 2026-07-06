"""Build the 200-image submission zip: forge each group's watermark onto its
clean targets under an LPIPS budget, then package a flat zip.

    python -m scripts.build_submission \
        --dataset data/Dataset --watermarks artifacts/wm.npz \
        --lpips-budget 0.04 --out submissions/budget040.zip

If ``--watermarks`` is omitted the estimates are extracted on the fly (auto
denoiser). Use ``--alpha`` instead of ``--lpips-budget`` for a fixed-scale,
decoder-free run that needs no LPIPS model. ``--budgets`` builds several zips in
one pass (one per budget) so a single job produces a sweep to submit hourly.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

from src import forge, quality, reembed
from src.data import iter_groups, load_group_sources, load_rgb, save_rgb
from src.extract import best_extraction


def load_watermarks(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files if not k.endswith("__meta")}


def get_watermarks(args) -> dict[str, np.ndarray]:
    if args.watermarks and Path(args.watermarks).exists():
        print(f"loading watermark estimates from {args.watermarks}")
        return load_watermarks(Path(args.watermarks))
    print("extracting watermark estimates on the fly (auto denoiser)")
    out = {}
    for g in iter_groups(args.dataset):
        ex = best_extraction(g.name, load_group_sources(g))
        out[g.name] = ex.watermark
        print(f"  {g.name}: denoiser={ex.denoiser} consistency={ex.consistency:.3f}")
    return out


def build_one(args, watermarks, budget, out_zip: Path, tmp_root: Path):
    tmp = tmp_root / out_zip.stem
    tmp.mkdir(parents=True, exist_ok=True)
    stats = []
    for g in iter_groups(args.dataset):
        scheme_cfg = None if args.no_reembed else reembed.GROUP_SCHEME.get(g.name)
        if scheme_cfg:
            # Identified public scheme: decode the shared message, re-embed it.
            method, length = scheme_cfg
            msg = reembed.extract_message(g.source_paths, method, length)
            for tpath in g.target_paths:
                clean = load_rgb(tpath)
                forged = reembed.embed_message(clean, msg, method)
                save_rgb(tmp / tpath.name, forged)
                stats.append((quality.lpips_distance(clean, forged, args.net),
                              quality.psnr(clean, forged), float("nan")))
            print(f"  {g.name}: re-embedded via {method} ({length}-bit)")
            continue
        delta = watermarks[g.name].astype(np.float32)
        g_budget = args.group_budgets.get(g.name, budget)
        for tpath in g.target_paths:
            clean = load_rgb(tpath)
            if args.alpha is not None:
                r = forge.forge_to_alpha(clean, delta, args.alpha)
            else:
                r = forge.forge_to_lpips(clean, delta, g_budget, net=args.net, cap_rms=args.cap_rms)
            save_rgb(tmp / tpath.name, r.forged)
            stats.append((r.lpips, r.psnr, r.alpha))
    # Override with images forged in a separate environment (e.g. VINE re-embed,
    # or FNNS/surrogate output) — any <name>.png present in a --prebuilt dir wins.
    for pb in args.prebuilt or []:
        pb = Path(pb)
        n = 0
        for f in pb.glob("*.png"):
            shutil.copy(f, tmp / f.name)
            n += 1
        print(f"  prebuilt override: {n} images from {pb}")

    pngs = sorted(tmp.glob("*.png"), key=lambda p: int(p.stem))
    assert len(pngs) == 200, f"expected 200 images, got {len(pngs)}"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=p.name)
    arr = np.array(stats, dtype=np.float64)
    lpips_vals = arr[:, 0][~np.isnan(arr[:, 0])]
    psnr_mean = arr[:, 1].mean()
    if lpips_vals.size:
        print(f"  -> {out_zip}  (200 imgs)  LPIPS mean={lpips_vals.mean():.4f} "
              f"max={lpips_vals.max():.4f}  PSNR mean={psnr_mean:.1f}dB")
    else:
        print(f"  -> {out_zip}  (200 imgs)  PSNR mean={psnr_mean:.1f}dB")


def main():
    ap = argparse.ArgumentParser(description="Forge watermarks onto targets and zip")
    ap.add_argument("--dataset", default="data/Dataset", type=Path)
    ap.add_argument("--watermarks", default="artifacts/wm.npz")
    ap.add_argument("--out", default="submissions/submission.zip", type=Path)
    ap.add_argument("--lpips-budget", type=float, default=0.04, dest="lpips_budget")
    ap.add_argument("--budgets", default=None,
                    help="comma-separated LPIPS budgets; builds one zip each")
    ap.add_argument("--alpha", type=float, default=None,
                    help="fixed scale instead of an LPIPS budget (no LPIPS model needed)")
    ap.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--cap-rms", type=float, default=24.0, dest="cap_rms")
    ap.add_argument("--no-reembed", action="store_true",
                    help="disable public-scheme re-embedding (additive transplant for all groups)")
    ap.add_argument("--prebuilt", action="append", default=[],
                    help="dir of pre-forged <name>.png images to override (repeatable); e.g. VINE WM_4 output")
    ap.add_argument("--group-budget", default="", dest="group_budget_str",
                    help="per-group LPIPS budget overrides, e.g. 'WM_3:0.04,WM_5:0.008'")
    ap.add_argument("--tmp", default="submissions/_tmp", type=Path)
    args = ap.parse_args()
    args.group_budgets = {}
    for part in filter(None, args.group_budget_str.split(",")):
        name, val = part.split(":")
        args.group_budgets[name.strip()] = float(val)

    watermarks = get_watermarks(args)
    if args.budgets:
        for b in [float(x) for x in args.budgets.split(",")]:
            out = args.out.with_name(f"{args.out.stem}_b{b:.3f}".replace(".", "") + ".zip")
            print(f"budget {b}:")
            build_one(args, watermarks, b, out, args.tmp)
    else:
        print(f"budget {args.lpips_budget}:")
        build_one(args, watermarks, args.lpips_budget, args.out, args.tmp)


if __name__ == "__main__":
    main()

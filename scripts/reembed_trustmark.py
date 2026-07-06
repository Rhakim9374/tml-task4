#!/usr/bin/env python
"""Forge WM_7 and WM_8 (Adobe TrustMark) watermarks by re-embedding the group's own secret.

WM_7 and WM_8 were identified as Adobe's TrustMark watermark (pip install trustmark==0.9.1).
The two groups use different TrustMark models / payload schemas:

  * WM_7 -> model_type 'Q', 100-bit backbone with BCH_5 error-correction (ECC).
    The 25 sources all carry the *same* ECC-validated 61-bit data payload
    (decode returns detected=True, version=1). We re-embed via the ECC schema.

  * WM_8 -> model_type 'P', 100-bit backbone with a RAW / non-ECC payload.
    The BCH decoder detects nothing (0/25), so we decode WITHOUT ECC to recover the
    raw 100-bit secret and re-embed it raw (use_ECC=False).

For each group this script:

  1. Decodes the group's 25 watermarked sources to the consensus secret via per-position
     majority vote (the ECC 61-bit payload for WM_7, the raw 100-bit string for WM_8) and
     reports cross-source agreement -- a sanity check that this is one shared watermark.
  2. Re-embeds that exact consensus secret into the group's 25 clean targets
     (WM_7 -> 151.png..175.png, WM_8 -> 176.png..200.png) with ``tm.encode`` using the
     SAME model_type / ECC setting.
  3. Saves all 50 forged PNGs, named by target id, to --out.
  4. Round-trip check: decodes a few forgeries under the correct model_type and reports
     bit-accuracy vs. the consensus secret plus LPIPS distortion vs. the clean target.

NUMPY >= 2 REQUIREMENT
----------------------
trustmark 0.9.1 pins ``numpy<2`` in its metadata but its code actually needs numpy>=2
(it uses ``np.long`` semantics / trips numpy 1.x ``RankWarning`` handling). Install it into
a FRESH venv, letting pip resolve trustmark first, then force-upgrade numpy:

    python3 -m venv tm_venv
    ./tm_venv/bin/pip install --upgrade pip
    ./tm_venv/bin/pip install "trustmark==0.9.1"
    ./tm_venv/bin/pip install --upgrade "numpy>=2"   # overrides trustmark's <2 pin (safe)
    ./tm_venv/bin/pip install lpips                  # for the LPIPS round-trip metric

TrustMark downloads its model checkpoints (Q and P) from Adobe on first use and caches them
inside the installed package. Run with that venv's python.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np


# ----------------------------------------------------------------------------------------
# Per-group configuration. sources_dir / target ids / model_type / ECC all differ per group.
# ----------------------------------------------------------------------------------------
GROUPS = {
    "WM_7": {
        "sources_subdir": "WM_7",
        "first_id": 151,
        "last_id": 175,
        "model_type": "Q",
        "use_ECC": True,      # BCH_5 ECC validates 25/25 for this group
    },
    "WM_8": {
        "sources_subdir": "WM_8",
        "first_id": 176,
        "last_id": 200,
        "model_type": "P",
        "use_ECC": False,     # raw 100-bit payload; BCH detects nothing -> decode raw
    },
}


def parse_args() -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sources-root",
        default=os.path.join(repo_root, "data/Dataset/watermarked_sources"),
        help="Root dir holding the WM_7/ and WM_8/ watermarked-source subfolders.",
    )
    p.add_argument(
        "--targets",
        default=os.path.join(repo_root, "data/Dataset/clean_targets"),
        help="Directory holding the clean targets 151.png..200.png.",
    )
    p.add_argument(
        "--out",
        default=os.path.join(repo_root, "artifacts/trustmark_wm78"),
        help="Output directory for the 50 forged PNGs (named by target id).",
    )
    p.add_argument(
        "--groups",
        default="WM_7,WM_8",
        help="Comma-separated groups to process (subset of WM_7,WM_8).",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    p.add_argument(
        "--n-verify",
        type=int,
        default=4,
        help="How many forgeries per group to round-trip-decode + LPIPS.",
    )
    return p.parse_args()


def majority_vote(bit_strings: list[str]) -> str:
    """Per-position majority vote over equal-length bit strings."""
    if not bit_strings:
        raise ValueError("no bit strings to vote on")
    n = len(bit_strings[0])
    if any(len(s) != n for s in bit_strings):
        raise ValueError("bit strings differ in length; cannot majority-vote")
    out = []
    for i in range(n):
        ones = sum(s[i] == "1" for s in bit_strings)
        out.append("1" if ones * 2 >= len(bit_strings) else "0")
    return "".join(out)


def bit_acc(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(a[i] == b[i] for i in range(n)) / n


def process_group(name, cfg, args, Image, TrustMark, lpips_fn, transforms):
    """Decode -> majority-vote -> re-embed -> save -> round-trip verify for one group."""
    from trustmark import TrustMark as _TM  # for Encoding enum

    print(f"\n{'='*78}\n[{name}] model_type={cfg['model_type']}  use_ECC={cfg['use_ECC']}")
    tm = TrustMark(
        verbose=False,
        device=args.device,
        model_type=cfg["model_type"],
        use_ECC=cfg["use_ECC"],
        encoding_type=_TM.Encoding.BCH_5,   # only consulted when use_ECC=True
    )

    sources_dir = os.path.join(args.sources_root, cfg["sources_subdir"])
    ids = list(range(cfg["first_id"], cfg["last_id"] + 1))

    # ----- 1. Decode the 25 sources to the consensus secret -----
    decoded, n_detected = [], 0
    for tid in ids:
        sp = os.path.join(sources_dir, f"src_{tid}.png")
        im = Image.open(sp).convert("RGB")
        secret, detected, version = tm.decode(im, MODE="binary")
        if detected and secret:
            decoded.append(secret)
            n_detected += 1
    if not decoded:
        raise SystemExit(f"[{name}] no source decoded a secret; aborting.")

    # Vote only over the modal length (robust if a stray source disagrees).
    modal_len = Counter(len(s) for s in decoded).most_common(1)[0][0]
    voting = [s for s in decoded if len(s) == modal_len]
    consensus = majority_vote(voting)
    agreement = float(np.mean([bit_acc(s, consensus) for s in voting]))
    print(f"[{name}] sources decoded/detected = {n_detected}/{len(ids)}  "
          f"secret_len={modal_len}  cross-source_agreement={agreement:.4f}")
    print(f"[{name}] consensus_secret = {consensus}")

    # ----- 2/3. Re-embed the consensus into the 25 clean targets and save -----
    os.makedirs(args.out, exist_ok=True)
    forged_paths: dict[int, str] = {}
    for tid in ids:
        cover = Image.open(os.path.join(args.targets, f"{tid}.png")).convert("RGB")
        stego = tm.encode(cover, consensus, MODE="binary")
        outp = os.path.join(args.out, f"{tid}.png")
        stego.save(outp)
        forged_paths[tid] = outp
    print(f"[{name}] wrote {len(forged_paths)} forged PNGs to {args.out}")

    # ----- 4. Round-trip verification (bit-accuracy + LPIPS) on a subset -----
    # Verify the first few plus the last id of the group.
    k = max(1, args.n_verify)
    verify_ids = ids[: k - 1] + [ids[-1]] if k > 1 else [ids[0]]
    verify_ids = sorted(set(verify_ids))
    accs, lps = [], []
    for tid in verify_ids:
        forged = Image.open(forged_paths[tid]).convert("RGB")
        secret, detected, version = tm.decode(forged, MODE="binary")
        acc = bit_acc(secret, consensus) if secret else 0.0
        accs.append(acc)
        line = f"[{name}] verify target {tid}: detected={detected} roundtrip_bitacc={acc:.4f}"
        if lpips_fn is not None:
            clean = Image.open(os.path.join(args.targets, f"{tid}.png")).convert("RGB")
            ct = transforms.ToTensor()(clean).unsqueeze(0).to(args.device) * 2 - 1
            ft = transforms.ToTensor()(forged).unsqueeze(0).to(args.device) * 2 - 1
            import torch
            with torch.no_grad():
                d = float(lpips_fn(ct, ft).item())
            lps.append(d)
            line += f" lpips={d:.4f}"
        print(line, flush=True)

    return {
        "name": name,
        "model_type": cfg["model_type"],
        "n_forged": len(forged_paths),
        "agreement": agreement,
        "consensus": consensus,
        "mean_bitacc": float(np.mean(accs)) if accs else None,
        "mean_lpips": float(np.mean(lps)) if lps else None,
        "n_verify": len(verify_ids),
    }


def main() -> int:
    args = parse_args()

    if int(np.__version__.split(".")[0]) < 2:
        print(f"[warn] numpy {np.__version__} < 2 detected; trustmark 0.9.1 needs numpy>=2. "
              f"Upgrade with: pip install --upgrade 'numpy>=2'", file=sys.stderr)

    from PIL import Image
    from torchvision import transforms
    from trustmark import TrustMark

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(args.device)
    except Exception as e:  # pragma: no cover
        print(f"[verify] LPIPS unavailable ({e}); reporting bit-accuracy only.")

    wanted = [g.strip() for g in args.groups.split(",") if g.strip()]
    results = []
    for name in wanted:
        if name not in GROUPS:
            raise SystemExit(f"unknown group {name!r}; choose from {list(GROUPS)}")
        results.append(
            process_group(name, GROUPS[name], args, Image, TrustMark, lpips_fn, transforms)
        )

    total = sum(r["n_forged"] for r in results)
    print(f"\n{'='*78}\n[summary] out={args.out}  total_forged={total}")
    for r in results:
        ba = "n/a" if r["mean_bitacc"] is None else f"{r['mean_bitacc']:.4f}"
        lp = "n/a" if r["mean_lpips"] is None else f"{r['mean_lpips']:.4f}"
        print(f"[summary] {r['name']} (model {r['model_type']}): forged={r['n_forged']} "
              f"agreement={r['agreement']:.4f} roundtrip_bitacc={ba} lpips={lp} "
              f"(n_verify={r['n_verify']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

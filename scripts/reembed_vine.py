#!/usr/bin/env python
"""Forge WM_4 (Meta VINE / VINE-R) watermarks by re-embedding the group's own message.

WM_4 was identified as Meta's VINE watermark (the VINE-R variant, Shilin-LU/VINE).
This script does an end-to-end, evaluator-native forgery:

  1. Load the public VINE-R decoder (Shilin-LU/VINE-R-Dec) and decode all 25 WM_4
     watermarked sources. Every source carries the *same* 100-bit payload, so we take
     the per-bit majority vote as the consensus message and report the cross-source
     agreement (a sanity check that this really is one shared watermark).
  2. Load the public VINE-R encoder (Shilin-LU/VINE-R-Enc) and re-embed that exact
     consensus message into each of the 25 clean targets 76.png..100.png.
  3. Save the 25 forged PNGs (named 76.png..100.png) to --out.
  4. Round-trip check: decode a few forgeries with VINE-R-Dec and report bit-accuracy
     against the consensus, plus LPIPS distortion vs. the clean target.

The upstream VINE code hardcodes ``.cuda()`` in a few places. On ``--device cpu`` we
monkeypatch those to no-ops so the whole pipeline runs on CPU; on ``--device cuda`` the
original CUDA calls are left intact.

Reproduce the environment (isolated venv + weights) with the commands in the module
docstring at the bottom / the task write-up.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sources",
        default=os.path.join(repo_root, "data/Dataset/watermarked_sources/WM_4"),
        help="Directory of the 25 WM_4 watermarked source PNGs.",
    )
    p.add_argument(
        "--targets",
        default=os.path.join(repo_root, "data/Dataset/clean_targets"),
        help="Directory holding the clean targets 76.png..100.png.",
    )
    p.add_argument(
        "--out",
        default=os.path.join(repo_root, "artifacts/vine_wm4"),
        help="Output directory for the 25 forged PNGs (named 76.png..100.png).",
    )
    p.add_argument(
        "--vine-repo",
        default=os.environ.get("VINE_REPO", "/private/tmp/VINE_repo"),
        help="Path to the cloned Shilin-LU/VINE repository (provides vine/src + diffusers fork).",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    p.add_argument("--enc-model", default="Shilin-LU/VINE-R-Enc", help="HF id / path of the VINE encoder.")
    p.add_argument("--dec-model", default="Shilin-LU/VINE-R-Dec", help="HF id / path of the VINE decoder.")
    p.add_argument(
        "--consensus",
        default=None,
        help="Optional 100-char bit-string to force the payload instead of decoding it from the sources.",
    )
    p.add_argument(
        "--verify-ids",
        default="76,77,78,100",
        help="Comma-separated target ids to round-trip-decode + LPIPS after forging.",
    )
    p.add_argument("--first-id", type=int, default=76)
    p.add_argument("--last-id", type=int, default=100)
    return p.parse_args()


def neutralize_cuda() -> None:
    """Make hardcoded ``.cuda()`` calls in the VINE code no-ops so it runs on CPU."""
    import torch

    torch.nn.Module.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
    if not torch.cuda.is_available():
        torch.cuda.empty_cache = lambda *a, **k: None  # type: ignore[assignment]


def crop_to_square(image):
    w, h = image.size
    m = min(w, h)
    l = (w - m) // 2
    t = (h - m) // 2
    return image.crop((l, t, l + m, t + m))


def main() -> int:
    args = parse_args()

    import torch  # noqa: F401  (import before neutralize so torch exists)

    if args.device == "cpu":
        neutralize_cuda()

    # Make `vine.src...` and bare `stega_encoder_decoder` / `vine_turbo` imports resolve.
    sys.path.insert(0, args.vine_repo)
    sys.path.insert(0, os.path.join(args.vine_repo, "vine", "src"))

    import torch
    from PIL import Image
    from torchvision import transforms

    from stega_encoder_decoder import CustomConvNeXt
    from vine_turbo import VINE_Turbo

    device = torch.device(args.device)

    to_256 = transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()]
    )

    # ----- 1. Decode the WM_4 sources to the consensus 100-bit message -----
    print(f"[decode] loading decoder {args.dec_model} on {device} ...", flush=True)
    decoder = CustomConvNeXt.from_pretrained(args.dec_model)
    decoder.to(device).eval()

    src_files = sorted(glob.glob(os.path.join(args.sources, "*.png")))
    if not src_files:
        raise SystemExit(f"No source PNGs found in {args.sources}")
    rows = []
    for f in src_files:
        x = to_256(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            rows.append((decoder(x)[0].cpu().numpy() > 0.5).astype(int))
    rows = np.array(rows)  # (25, 100)
    decoded_consensus = (rows.mean(0) > 0.5).astype(int)
    agreement = float((rows == decoded_consensus).mean())
    balance = float(decoded_consensus.mean())
    decoded_str = "".join(map(str, decoded_consensus))
    print(f"[decode] sources={len(src_files)} agreement={agreement:.4f} balance={balance:.4f}")
    print(f"[decode] consensus={decoded_str}")

    if args.consensus is not None:
        if len(args.consensus) != len(decoded_consensus):
            raise SystemExit(f"--consensus must be {len(decoded_consensus)} bits, got {len(args.consensus)}")
        wm_bits = np.array([int(c) for c in args.consensus])
        if decoded_str != args.consensus:
            print(f"[decode] NOTE: forcing --consensus, differs from decoded in "
                  f"{int((wm_bits != decoded_consensus).sum())} bit(s).")
    else:
        wm_bits = decoded_consensus
    watermark = torch.tensor(wm_bits, dtype=torch.float).unsqueeze(0).to(device)

    # ----- 2/3. Re-embed into all 25 clean targets and save -----
    print(f"[embed] loading encoder {args.enc_model} on {device} ...", flush=True)
    enc = VINE_Turbo.from_pretrained(args.enc_model, device=args.device)
    enc.to(device).eval()

    os.makedirs(args.out, exist_ok=True)
    ids = list(range(args.first_id, args.last_id + 1))
    forged_paths: dict[int, str] = {}
    for tid in ids:
        p = os.path.join(args.targets, f"{tid}.png")
        pil = Image.open(p).convert("RGB")
        if pil.size[0] != pil.size[1]:
            pil = crop_to_square(pil)
        size = pil.size  # (w, h) of the (possibly cropped) original

        resized = (2.0 * to_256(pil) - 1.0).unsqueeze(0).to(device)
        full = (2.0 * transforms.ToTensor()(pil) - 1.0).unsqueeze(0).to(device)
        up = transforms.Resize(size[::-1], interpolation=transforms.InterpolationMode.BICUBIC)
        with torch.no_grad():
            enc256 = enc(resized, watermark)
            residual_full = up(enc256 - resized)  # residual at native resolution
            encoded = torch.clamp((residual_full + full) * 0.5 + 0.5, 0.0, 1.0)
        out_pil = transforms.ToPILImage()(encoded[0].cpu())
        sp = os.path.join(args.out, f"{tid}.png")
        out_pil.save(sp)
        forged_paths[tid] = sp
    print(f"[embed] wrote {len(forged_paths)} forged PNGs to {args.out}")

    # ----- 4. Round-trip verification (bit-accuracy + LPIPS) on a subset -----
    try:
        import lpips

        loss_fn = lpips.LPIPS(net="alex").to(device)
        has_lpips = True
    except Exception as e:  # pragma: no cover
        print(f"[verify] LPIPS unavailable ({e}); reporting bit-accuracy only.")
        has_lpips = False

    verify_ids = [int(s) for s in args.verify_ids.split(",") if s.strip()]
    accs, lps = [], []
    for tid in verify_ids:
        sp = forged_paths[tid]
        forged_pil = Image.open(sp).convert("RGB")
        x = to_256(forged_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = (decoder(x)[0].cpu().numpy() > 0.5).astype(int)
        acc = float((pred == wm_bits).mean())
        accs.append(acc)
        line = f"[verify] target {tid}: roundtrip_bitacc={acc:.4f}"
        if has_lpips:
            clean_pil = Image.open(os.path.join(args.targets, f"{tid}.png")).convert("RGB")
            if clean_pil.size[0] != clean_pil.size[1]:
                clean_pil = crop_to_square(clean_pil)
            clean_t = (transforms.ToTensor()(clean_pil).unsqueeze(0).to(device) * 2 - 1)
            forged_t = (transforms.ToTensor()(forged_pil).unsqueeze(0).to(device) * 2 - 1)
            with torch.no_grad():
                d = float(loss_fn(clean_t, forged_t).item())
            lps.append(d)
            line += f" lpips={d:.4f}"
        print(line, flush=True)

    print(f"\n[summary] forged={len(forged_paths)} out={args.out}")
    print(f"[summary] decode agreement={agreement:.4f} balance={balance:.4f}")
    if accs:
        print(f"[summary] mean roundtrip_bitacc={np.mean(accs):.4f} (n={len(accs)})")
    if lps:
        print(f"[summary] mean lpips={np.mean(lps):.4f} (n={len(lps)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

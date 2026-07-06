#!/usr/bin/env python
"""Forge WM_6 (MBRS) watermarks by re-embedding the group's own message.

WM_6 was identified as the MBRS watermark (jzyustc/MBRS), specifically the
pretrained ``MBRS_256_256`` model (256x256 images, 256-bit message, checkpoint
``EC_42.pth``). This script does an end-to-end, evaluator-native forgery:

  1. Load the MBRS EncoderDecoder (Encoder_MP + Decoder) with the EC_42.pth weights
     and decode all 25 WM_6 watermarked sources. Every source carries the *same*
     256-bit payload, so we take the per-bit majority vote as the consensus message
     and report the cross-source agreement (a sanity check that this really is one
     shared watermark).
  2. Re-embed that exact consensus message into each of the 25 clean targets
     126.png..150.png with the MBRS encoder.
  3. Save the 25 forged PNGs (named 126.png..150.png) to --out.
  4. Round-trip check: decode a few forgeries with the MBRS decoder and report
     bit-accuracy against the consensus, plus LPIPS distortion vs. the clean target.

All images are 256x256 RGB (MBRS native resolution), so no cropping/resizing of the
targets is needed. Runs on CPU by default; pass --device cuda on the GPU cluster.

Environment reproduction (isolated venv + weights) -- see the commands at the bottom
of this docstring / the task write-up:

    python3 -m venv /tmp/mbrs_venv
    source /tmp/mbrs_venv/bin/activate
    pip install --upgrade pip
    pip install "numpy<2" pillow torch torchvision lpips
    pip install gdown
    git clone https://github.com/jzyustc/MBRS.git /tmp/MBRS
    # Pretrained MBRS_256_256 weights live in the repo's Google Drive folder
    #   https://drive.google.com/drive/folders/1A_SAqvU2vMsHxki0s9m9rKa-g8B6aghe
    # (linked from the MBRS README "Pretrained Models" section). Download the
    # MBRS_256_256 folder (contains EC_42.pth, the encoder-decoder checkpoint):
    mkdir -p /tmp/MBRS/dl
    gdown --folder 1A_SAqvU2vMsHxki0s9m9rKa-g8B6aghe -O /tmp/MBRS/dl
    # -> yields /tmp/MBRS/dl/MBRS_256_256/EC_42.pth
    python scripts/reembed_mbrs.py --mbrs-repo /tmp/MBRS \
        --weights /tmp/MBRS/dl/MBRS_256_256/EC_42.pth --device cpu
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sources",
        default=os.path.join(repo_root, "data/Dataset/watermarked_sources/WM_6"),
        help="Directory of the 25 WM_6 watermarked source PNGs.",
    )
    p.add_argument(
        "--targets",
        default=os.path.join(repo_root, "data/Dataset/clean_targets"),
        help="Directory holding the clean targets 126.png..150.png.",
    )
    p.add_argument(
        "--out",
        default=os.path.join(repo_root, "artifacts/mbrs_wm6"),
        help="Output directory for the 25 forged PNGs (named 126.png..150.png).",
    )
    p.add_argument(
        "--mbrs-repo",
        default=os.environ.get("MBRS_REPO", "/private/tmp/MBRS"),
        help="Path to the cloned jzyustc/MBRS repository (provides the network/ package).",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Path to the EC_42.pth EncoderDecoder checkpoint "
        "(default: <mbrs-repo>/dl/MBRS_256_256/EC_42.pth).",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    p.add_argument("--H", type=int, default=256, help="Image height MBRS was trained on.")
    p.add_argument("--W", type=int, default=256, help="Image width MBRS was trained on.")
    p.add_argument("--message-length", type=int, default=256, help="MBRS payload length in bits.")
    p.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Residual strength: encoded = clean + strength * (encoder(clean,msg) - clean).",
    )
    p.add_argument(
        "--consensus",
        default=None,
        help="Optional 256-char bit-string to force the payload instead of decoding it from sources.",
    )
    p.add_argument(
        "--verify-ids",
        default="126,127,128,150",
        help="Comma-separated target ids to round-trip-decode + LPIPS after forging.",
    )
    p.add_argument("--first-id", type=int, default=126)
    p.add_argument("--last-id", type=int, default=150)
    return p.parse_args()


def load_img(path: str, size: int, device) -> "torch.Tensor":
    """Load a PNG as a [-1,1] tensor at (size,size), matching MBRS test preprocessing."""
    import torch
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    return ((t - 0.5) / 0.5).to(device)


def to_pil(tensor):
    """Convert a single [-1,1] CHW tensor back to a PIL RGB image."""
    from PIL import Image

    arr = tensor.detach().cpu().clamp(-1, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(np.round((arr * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> int:
    args = parse_args()

    import torch
    from PIL import Image  # noqa: F401  (used indirectly)

    # Make `from network... import ...` resolve against the cloned MBRS repo.
    sys.path.insert(0, args.mbrs_repo)
    from network.Encoder_MP_Decoder import EncoderDecoder

    device = torch.device(args.device)
    weights = args.weights or os.path.join(args.mbrs_repo, "dl/MBRS_256_256/EC_42.pth")
    if not os.path.isfile(weights):
        raise SystemExit(f"Weights not found: {weights}")

    size = args.H
    if args.H != args.W:
        raise SystemExit("This forgery assumes square MBRS input (H == W).")

    # ----- build MBRS EncoderDecoder (Identity noise layer, inference only) -----
    print(f"[load] EncoderDecoder(H={args.H},W={args.W},ml={args.message_length}) on {device}", flush=True)
    ed = EncoderDecoder(args.H, args.W, args.message_length, ["Identity()"]).to(device)
    ed.load_state_dict(torch.load(weights, map_location=device))
    ed.eval()

    # ----- 1. Decode WM_6 sources to the consensus 256-bit message -----
    src_files = sorted(
        glob.glob(os.path.join(args.sources, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
    )
    if not src_files:
        raise SystemExit(f"No source PNGs found in {args.sources}")
    rows = []
    for f in src_files:
        t = load_img(f, size, device)
        with torch.no_grad():
            out = ed.decoder(t)
        rows.append((out.gt(0.5)).int().cpu().numpy().reshape(-1))
    rows = np.stack(rows)  # (25, ml)
    decoded_consensus = (rows.mean(0) > 0.5).astype(int)
    agreement = float((rows == decoded_consensus[None, :]).mean())
    balance = float(decoded_consensus.mean())
    decoded_str = "".join(map(str, decoded_consensus))
    print(f"[decode] sources={len(src_files)} agreement={agreement:.4f} "
          f"balance={balance:.4f} ones={int(decoded_consensus.sum())}/{args.message_length}")
    print(f"[decode] consensus={decoded_str}")

    if args.consensus is not None:
        if len(args.consensus) != args.message_length:
            raise SystemExit(f"--consensus must be {args.message_length} bits, got {len(args.consensus)}")
        wm_bits = np.array([int(c) for c in args.consensus])
        if decoded_str != args.consensus:
            print(f"[decode] NOTE: forcing --consensus, differs from decoded in "
                  f"{int((wm_bits != decoded_consensus).sum())} bit(s).")
    else:
        wm_bits = decoded_consensus
    msg = torch.from_numpy(wm_bits.astype(np.float32)).unsqueeze(0).to(device)

    # ----- 2/3. Re-embed into all 25 clean targets and save -----
    os.makedirs(args.out, exist_ok=True)
    ids = list(range(args.first_id, args.last_id + 1))
    forged_paths: dict[int, str] = {}
    for tid in ids:
        clean = load_img(os.path.join(args.targets, f"{tid}.png"), size, device)
        with torch.no_grad():
            enc = ed.encoder(clean, msg)
            encoded = clean + (enc - clean) * args.strength
        sp = os.path.join(args.out, f"{tid}.png")
        to_pil(encoded[0]).save(sp)
        forged_paths[tid] = sp
    print(f"[embed] wrote {len(forged_paths)} forged PNGs to {args.out}")

    # ----- 4. Round-trip verification (bit-accuracy + LPIPS) on a subset -----
    try:
        import lpips

        loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        has_lpips = True
    except Exception as e:  # pragma: no cover
        print(f"[verify] LPIPS unavailable ({e}); reporting bit-accuracy only.")
        has_lpips = False

    verify_ids = [int(s) for s in args.verify_ids.split(",") if s.strip()]
    accs, lps = [], []
    for tid in verify_ids:
        forged = load_img(forged_paths[tid], size, device)  # reload the saved PNG (true round-trip)
        with torch.no_grad():
            pred = (ed.decoder(forged).gt(0.5)).int().cpu().numpy().reshape(-1)
        acc = float((pred == wm_bits).mean())
        accs.append(acc)
        line = f"[verify] target {tid}: roundtrip_bitacc={acc:.4f}"
        if has_lpips:
            clean = load_img(os.path.join(args.targets, f"{tid}.png"), size, device)
            with torch.no_grad():
                d = float(loss_fn(clean.clamp(-1, 1), forged.clamp(-1, 1)).item())
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

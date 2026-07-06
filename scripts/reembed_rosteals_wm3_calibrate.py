#!/usr/bin/env python
"""Calibrate the genuine RoSteALS re-embed for WM_3 to sit under the LPIPS budget, and
report a properly-leashed (non-adversarial) FNNS as a comparison.

The bare re-embed is a genuine same-family watermark (proxy bit-acc ~0.997) but lands at
LPIPS ~0.032, marginally over the 0.03 budget. We scale the re-embed *residual* (a genuine
RoSteALS watermark pattern) by s<=1 and keep the largest s whose mean LPIPS <= budget --
this preserves watermark strength while fitting the budget, and unlike an FNNS proxy-only
artifact it should transfer to the hidden decoder. We also run FNNS with a strong L2 leash
toward the re-embed so it stays a plausible watermark (not adversarial noise).
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, torch

ROOT = "/Users/rafihakim/Desktop/SaarlandBachelorsCode/Semester6/tml-task4"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default=f"{ROOT}/data/Dataset/watermarked_sources/WM_3")
    p.add_argument("--targets", default=f"{ROOT}/data/Dataset/clean_targets")
    p.add_argument("--out", default=f"{ROOT}/artifacts/wm3_proxy")
    p.add_argument("--config", default="models/VQ4_mir_inference.yaml")
    p.add_argument("--weight", default="models/RoSteALS/epoch=000017-step=000449999.ckpt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--budget", type=float, default=0.03)
    return p.parse_args()


def load_u8(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BILINEAR)
    return np.array(im).astype(np.float32)


def to_pm1(u8, device):
    return torch.from_numpy(u8 / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).float().to(device)


def save_u8(u8, path):
    from PIL import Image
    Image.fromarray(np.clip(u8, 0, 255).astype(np.uint8)).save(path)


def main():
    args = parse_args()
    os.chdir("/private/tmp/RoSteALS"); sys.path.insert(0, "/private/tmp/RoSteALS")
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    import lpips

    device = args.device
    lp = lpips.LPIPS(net="alex").to(device)
    for p in lp.parameters(): p.requires_grad_(False)

    cfg = OmegaConf.load(args.config).model
    if "noise_config" in cfg.params: del cfg.params["noise_config"]
    cfg.params.decoder_config.params.secret_len = cfg.params.control_config.params.secret_len
    model = instantiate_from_config(cfg)
    sd = torch.load(args.weight, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    model.load_state_dict(sd, strict=False); model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)

    def bits(x): return (model.decoder(x) > 0).int().cpu().numpy()[0]

    # consensus
    rows = [bits(to_pm1(load_u8(f), device)) for f in sorted(glob.glob(f"{args.sources}/*.png"))]
    rows = np.array(rows); consensus = (rows.mean(0) > 0.5).astype(int)
    consensus_t = torch.from_numpy(consensus.astype(np.float32)).unsqueeze(0).to(device)
    print(f"[decode] agreement={float((rows==consensus).mean()):.4f} balance={consensus.mean():.4f}")

    ids = list(range(51, 76))
    clean = {i: to_pm1(load_u8(f"{args.targets}/{i}.png"), device) for i in ids}

    # ---- re-embed: capture genuine residual per image ----
    res = {}
    for i in ids:
        with torch.no_grad():
            z = model.encode_first_stage(clean[i])
            z_e, _ = model(z, None, consensus_t)
            stego = model.decode_first_stage(z_e).clamp(-1, 1)
        res[i] = (stego - clean[i])            # genuine RoSteALS watermark residual, [-1,1] scale

    # ---- scale sweep on the genuine residual ----
    print("[reembed-scale]  s     bitacc   lpips")
    best = None
    for s in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]:
        accs, lps = [], []
        for i in ids:
            forged = (clean[i] + s * res[i]).clamp(-1, 1)
            accs.append(float((bits(forged) == consensus).mean()))
            with torch.no_grad(): lps.append(float(lp(clean[i], forged).item()))
        ma, ml = float(np.mean(accs)), float(np.mean(lps))
        print(f"[reembed-scale] {s:.2f}   {ma:.4f}   {ml:.4f}")
        if ml <= args.budget and best is None:      # largest s (first) under budget
            best = (s, ma, ml)
    if best is None:
        best = (0.7, ma, ml)
    s_best, acc_best, lp_best = best
    print(f"[reembed-scale] CHOSEN s={s_best} bitacc={acc_best:.4f} lpips={lp_best:.4f}")

    # save the calibrated genuine re-embed
    os.makedirs(args.out, exist_ok=True)
    for i in ids:
        forged = (clean[i] + s_best * res[i]).clamp(-1, 1)
        u8 = (forged[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        save_u8(u8, f"{args.out}/{i}.png")
    print(f"[save] wrote 25 calibrated re-embed forgeries to {args.out}")

    # ---- FNNS with a STRONG leash toward the genuine re-embed (non-adversarial) ----
    bce = torch.nn.BCEWithLogitsLoss()
    accs, lps = [], []
    for i in ids:
        target_res = (s_best * res[i]).detach()
        delta = target_res.clone().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=0.005)
        for _ in range(80):
            opt.zero_grad()
            forged = (clean[i] + delta).clamp(-1, 1)
            loss = (bce(model.decoder(forged), consensus_t)
                    + 6.0 * lp(clean[i], forged).mean()
                    + 8.0 * ((delta - target_res) ** 2).mean())   # strong leash -> stays a watermark
            loss.backward(); opt.step()
        with torch.no_grad(): forged = (clean[i] + delta).clamp(-1, 1)
        accs.append(float((bits(forged) == consensus).mean()))
        with torch.no_grad(): lps.append(float(lp(clean[i], forged).item()))
    print(f"[fnns-strong-leash] bitacc={np.mean(accs):.4f} lpips={np.mean(lps):.4f}")

    print("RESULT_JSON " + repr({
        "reembed_scale_chosen": s_best, "reembed_bitacc": acc_best, "reembed_lpips": lp_best,
        "fnns_leashed_bitacc": float(np.mean(accs)), "fnns_leashed_lpips": float(np.mean(lps)),
        "out": args.out,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Forge WM_3 using RoSteALS (TuBui/RoSteALS, VQ-f4, 100-bit) as a same-family PROXY.

WM_3 is a non-public learned CNN watermark. No public decoder reads it cleanly, but the
public RoSteALS decoder reads the WM_3 sources at ~0.80 cross-source agreement (well above
the ~0.59 clean floor) -- a genuine partial signal from a same-family model. We exploit
that proxy three ways and keep the set with the highest PROXY bit-accuracy under an LPIPS
budget. We CANNOT measure the true (hidden) decoder locally; proxy bit-acc is only a guide,
the real test is the leaderboard.

  (1) TRANSPLANT : w_hat = mean_i(src_i - nlmeans(src_i)); forged = clean + alpha*w_hat,
                   alpha swept to maximize proxy bit-acc (calibrates amplitude).
  (2) RE-EMBED   : decode the consensus 100-bit message from the sources with the RoSteALS
                   decoder, re-embed it into the clean targets with the RoSteALS encoder.
  (3) FNNS       : warm-start delta = best transplant, Adam-optimize delta to make
                   proxy_decoder(clean+delta) == consensus, with an LPIPS penalty and an
                   L2 leash toward the transplant (stays a plausible watermark).

Run inside the isolated RoSteALS venv from the RoSteALS repo dir:
  cd /private/tmp/RoSteALS
  /private/tmp/wf4_rosteals_env/bin/python <thisfile> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = "/Users/rafihakim/Desktop/SaarlandBachelorsCode/Semester6/tml-task4"
    p.add_argument("--sources", default=f"{root}/data/Dataset/watermarked_sources/WM_3")
    p.add_argument("--targets", default=f"{root}/data/Dataset/clean_targets")
    p.add_argument("--out", default=f"{root}/artifacts/wm3_proxy")
    p.add_argument("--rosteals-repo", default="/private/tmp/RoSteALS")
    p.add_argument("--config", default="models/VQ4_mir_inference.yaml")
    p.add_argument("--weight", default="models/RoSteALS/epoch=000017-step=000449999.ckpt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--first-id", type=int, default=51)
    p.add_argument("--last-id", type=int, default=75)
    p.add_argument("--lpips-budget", type=float, default=0.03)
    p.add_argument("--fnns-steps", type=int, default=150)
    p.add_argument("--fnns-lr", type=float, default=0.01)
    p.add_argument("--fnns-lpips-w", type=float, default=6.0)
    p.add_argument("--fnns-leash-w", type=float, default=0.10)
    return p.parse_args()


# ------------------------- image helpers (256x256 RGB) ----------------------
def load_u8(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BILINEAR)
    return np.array(im).astype(np.float32)  # [0,255], (256,256,3)


def to_pm1_tensor(u8, device):
    # (H,W,3)[0,255] -> (1,3,H,W)[-1,1]
    t = torch.from_numpy(u8 / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).float()
    return t.to(device)


def save_u8(u8, path):
    from PIL import Image
    Image.fromarray(np.clip(u8, 0, 255).astype(np.uint8)).save(path)


def nlmeans(u8):
    from skimage.restoration import denoise_nl_means, estimate_sigma
    x = u8 / 255.0
    sigma = float(np.mean(estimate_sigma(x, channel_axis=-1)))
    den = denoise_nl_means(x, h=1.15 * sigma, sigma=sigma, fast_mode=True,
                           patch_size=5, patch_distance=6, channel_axis=-1)
    return den * 255.0


# ------------------------------- proxy model --------------------------------
def load_model(args):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    cfg = OmegaConf.load(args.config).model
    if "noise_config" in cfg.params:          # training-only; pulls in a slack dep
        del cfg.params["noise_config"]
    cfg.params.decoder_config.params.secret_len = cfg.params.control_config.params.secret_len
    model = instantiate_from_config(cfg)
    sd = torch.load(args.weight, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[model] loaded  missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def decode_bits(model, x_pm1):
    """logits -> hard bits (1,100). x_pm1 in [-1,1]."""
    logits = model.decoder(x_pm1)
    return (logits > 0).int().cpu().numpy()[0]


# ---------------------------------- main ------------------------------------
def main():
    args = parse_args()
    sys.path.insert(0, args.rosteals_repo)
    os.chdir(args.rosteals_repo)
    device = args.device
    torch.manual_seed(0)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    def lpips_d(clean_pm1, forged_pm1):
        with torch.no_grad():
            return float(lpips_fn(clean_pm1, forged_pm1).item())

    model = load_model(args)

    # ---- 1. decode consensus message from the 25 WM_3 sources ----
    src_files = sorted(glob.glob(os.path.join(args.sources, "*.png")))
    rows = []
    for f in src_files:
        with torch.no_grad():
            rows.append(decode_bits(model, to_pm1_tensor(load_u8(f), device)))
    rows = np.array(rows)                       # (25,100)
    consensus = (rows.mean(0) > 0.5).astype(int)
    agreement = float((rows == consensus).mean())
    balance = float(consensus.mean())
    print(f"[decode] sources={len(src_files)} agreement={agreement:.4f} balance={balance:.4f}")
    consensus_t = torch.from_numpy(consensus.astype(np.float32)).unsqueeze(0).to(device)  # (1,100)

    ids = list(range(args.first_id, args.last_id + 1))
    targets_u8 = {i: load_u8(os.path.join(args.targets, f"{i}.png")) for i in ids}
    targets_pm1 = {i: to_pm1_tensor(targets_u8[i], device) for i in ids}

    def proxy_bitacc(forged_pm1):
        return float((decode_bits(model, forged_pm1) == consensus).mean())

    # ---- clean floor: proxy bit-acc on the untouched targets ----
    clean_acc = float(np.mean([proxy_bitacc(targets_pm1[i]) for i in ids]))
    print(f"[floor] proxy bit-acc on CLEAN targets = {clean_acc:.4f}")

    # =================== Method 1: TRANSPLANT ===================
    print("[transplant] estimating w_hat = mean_i(src - nlmeans(src)) over 25 sources ...", flush=True)
    resid = np.zeros((256, 256, 3), dtype=np.float64)
    for f in src_files:
        u8 = load_u8(f)
        resid += (u8 - nlmeans(u8))
    w_hat = (resid / len(src_files)).astype(np.float32)   # [0,255]-scale residual
    print(f"[transplant] w_hat std={w_hat.std():.4f} absmax={np.abs(w_hat).max():.4f}")

    alphas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 26.0, 32.0]
    best_alpha, best_acc = 0.0, -1.0
    sweep = []
    for a in alphas:
        accs, lps = [], []
        for i in ids:
            forged = targets_u8[i] + a * w_hat
            fp = to_pm1_tensor(forged, device)
            accs.append(proxy_bitacc(fp))
            lps.append(lpips_d(targets_pm1[i], fp))
        ma, ml = float(np.mean(accs)), float(np.mean(lps))
        sweep.append((a, ma, ml))
        print(f"[transplant] alpha={a:5.1f}  bitacc={ma:.4f}  lpips={ml:.4f}")
        # pick the amplitude that maximizes proxy bit-acc while staying under budget
        if ml <= args.lpips_budget and ma > best_acc:
            best_acc, best_alpha = ma, a
    # fallback if nothing under budget (shouldn't happen): take max-acc overall
    if best_acc < 0:
        best_alpha, best_acc, _ = max(sweep, key=lambda t: t[1])
    transplant = {}
    t_accs, t_lps = [], []
    for i in ids:
        forged = targets_u8[i] + best_alpha * w_hat
        transplant[i] = np.clip(forged, 0, 255)
        fp = to_pm1_tensor(forged, device)
        t_accs.append(proxy_bitacc(fp))
        t_lps.append(lpips_d(targets_pm1[i], fp))
    transplant_acc, transplant_lpips = float(np.mean(t_accs)), float(np.mean(t_lps))
    print(f"[transplant] BEST alpha={best_alpha}  bitacc={transplant_acc:.4f}  lpips={transplant_lpips:.4f}")

    # =================== Method 2: RE-EMBED ===================
    print("[reembed] embedding consensus with the RoSteALS encoder ...", flush=True)
    reembed = {}
    r_accs, r_lps = [], []
    for i in ids:
        cover = targets_pm1[i]
        with torch.no_grad():
            z = model.encode_first_stage(cover)
            z_embed, _ = model(z, None, consensus_t)
            stego = model.decode_first_stage(z_embed).clamp(-1, 1)
        fp = stego
        u8 = ((stego[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
        reembed[i] = np.clip(u8, 0, 255)
        r_accs.append(proxy_bitacc(fp))
        r_lps.append(lpips_d(cover, fp))
    reembed_acc, reembed_lpips = float(np.mean(r_accs)), float(np.mean(r_lps))
    print(f"[reembed] bitacc={reembed_acc:.4f}  lpips={reembed_lpips:.4f}")

    # =================== Method 3: FNNS ===================
    print(f"[fnns] refining delta (steps={args.fnns_steps} lr={args.fnns_lr} "
          f"lpips_w={args.fnns_lpips_w} leash_w={args.fnns_leash_w}) ...", flush=True)
    bce = torch.nn.BCEWithLogitsLoss()
    fnns = {}
    f_accs, f_lps = [], []
    for i in ids:
        clean = targets_pm1[i]                                   # (1,3,256,256) [-1,1]
        delta0 = torch.from_numpy((best_alpha * w_hat) / 127.5)  # transplant in [-1,1] scale
        delta0 = delta0.permute(2, 0, 1).unsqueeze(0).float().to(device)
        delta = delta0.clone().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=args.fnns_lr)
        for _ in range(args.fnns_steps):
            opt.zero_grad()
            forged = (clean + delta).clamp(-1, 1)
            logits = model.decoder(forged)
            loss = (bce(logits, consensus_t)
                    + args.fnns_lpips_w * lpips_fn(clean, forged).mean()
                    + args.fnns_leash_w * ((delta - delta0) ** 2).mean())
            loss.backward()
            opt.step()
        with torch.no_grad():
            forged = (clean + delta).clamp(-1, 1)
        u8 = ((forged[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
        fnns[i] = np.clip(u8, 0, 255)
        f_accs.append(proxy_bitacc(forged))
        f_lps.append(lpips_d(clean, forged))
    fnns_acc, fnns_lpips = float(np.mean(f_accs)), float(np.mean(f_lps))
    print(f"[fnns] bitacc={fnns_acc:.4f}  lpips={fnns_lpips:.4f}")

    # =================== choose best under LPIPS budget ===================
    methods = {
        "transplant": (transplant_acc, transplant_lpips, transplant),
        "reembed":    (reembed_acc, reembed_lpips, reembed),
        "fnns":       (fnns_acc, fnns_lpips, fnns),
    }
    print("\n[summary] method            bitacc   lpips   under_budget")
    for name, (acc, lp, _) in methods.items():
        print(f"[summary] {name:16s}  {acc:.4f}  {lp:.4f}   {lp <= args.lpips_budget}")

    eligible = {k: v for k, v in methods.items() if v[1] <= args.lpips_budget}
    pool = eligible if eligible else methods
    chosen = max(pool.items(), key=lambda kv: kv[1][0])
    chosen_name, (chosen_acc, chosen_lpips, chosen_imgs) = chosen

    os.makedirs(args.out, exist_ok=True)
    for i in ids:
        save_u8(chosen_imgs[i], os.path.join(args.out, f"{i}.png"))
    print(f"\n[save] chosen={chosen_name} bitacc={chosen_acc:.4f} lpips={chosen_lpips:.4f}")
    print(f"[save] wrote {len(ids)} forgeries to {args.out}")

    # machine-readable last line
    print("RESULT_JSON " + repr({
        "agreement": agreement, "balance": balance, "clean_floor": clean_acc,
        "transplant": {"alpha": best_alpha, "bitacc": transplant_acc, "lpips": transplant_lpips},
        "reembed": {"bitacc": reembed_acc, "lpips": reembed_lpips},
        "fnns": {"bitacc": fnns_acc, "lpips": fnns_lpips},
        "chosen": chosen_name, "chosen_bitacc": chosen_acc, "chosen_lpips": chosen_lpips,
        "n_forged": len(ids), "out": args.out,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

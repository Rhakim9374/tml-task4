"""GPU FNNS forgery of WM_3 (RoSteALS solo or RoSteALS+PIMoG ensemble) with a built-in
held-out cross-decoder judge. One condor process = one strategy/hyperparameter variant;
queue many in wm3.sub to sweep across GPUs in parallel.

Paths come from $REPO (the cluster checkout root). Models are fetched by fetch_models.sh into
$REPO/artifacts/wm3_models/. Writes forged PNGs 51..75 to --outdir and a scorecard.json next to it.

Why FNNS-solo tends to fail: optimizing only against RoSteALS makes a RoSteALS-specific
(adversarial) mark that a HELD-OUT decoder (PIMoG) can't read -> won't transfer to the evaluator.
The scorecard reports the held-out PIMoG reading so we submit only a candidate that generalizes.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

REPO = os.environ.get("REPO", os.getcwd())
DATA = f"{REPO}/data/Dataset"
MODELS = f"{REPO}/artifacts/wm3_models"
sys.path.insert(0, f"{MODELS}/PIMoG")          # for `from model import Encoder_Decoder`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDS = list(range(51, 76))


# ---- checkpoint loader that tolerates the RoSteALS Lightning checkpoint w/o pulling PL ----
def load_ckpt(path):
    import types
    for name in ("pytorch_lightning", "torchmetrics"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            if name == "pytorch_lightning":
                m.LightningModule = nn.Module
            sys.modules[name] = m
    return torch.load(path, map_location="cpu", weights_only=False)


def build_rosteals():
    sd = load_ckpt(f"{MODELS}/rosteals_control.ckpt")
    state = sd.get("state_dict", sd)
    pre = "decoder.decoder."
    dec = {k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}
    net = torchvision.models.resnet50(weights=None)
    net.fc = nn.Linear(2048, 100)
    net.load_state_dict(dec, strict=False)
    net.eval().to(DEVICE)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def build_pimog():
    from model import Encoder_Decoder
    net = Encoder_Decoder("Identity")
    raw = torch.load(f"{MODELS}/pimog.pth", map_location="cpu")
    psd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in raw.items()}
    net.load_state_dict(psd, strict=False)
    P = net.Decoder.eval().to(DEVICE)
    for p in P.parameters():
        p.requires_grad_(False)
    return P


def logits_R(R, x01):
    return R(x01 * 2 - 1)


def logits_P(P, x01):
    bgr = x01[:, [2, 1, 0], :, :]
    b128 = F.interpolate(bgr, size=(128, 128), mode="bilinear", align_corners=False)
    return P(b128 * 2 - 1)


# ---- exact (non-diff) verification decoders, used by the judge on saved uint8 PNGs ----
_tform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])


@torch.no_grad()
def verify_R(R, img_bgr):
    rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
    x = _tform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    return (R(x) > 0).cpu().numpy().reshape(-1).astype(int)


@torch.no_grad()
def verify_P(P, img_bgr):
    im = cv2.resize(img_bgr, (128, 128)).transpose((2, 0, 1))
    t = torch.from_numpy(np.float32(im / 255.0 * 2 - 1)).unsqueeze(0).to(DEVICE)
    return P(t).cpu().numpy().round().clip(0, 1).reshape(-1).astype(int)


def quantize_st(x):
    q = torch.round(x.clamp(0, 1) * 255) / 255
    return x + (q - x).detach()


def jpeg_st(x, q):
    out = torch.empty_like(x)
    xn = x.clamp(0, 1).detach().cpu().numpy()
    for i in range(x.shape[0]):
        rgb = (xn[i].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        _, enc = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
        out[i] = torch.from_numpy(np.ascontiguousarray(dec.transpose(2, 0, 1) / 255.0)).float()
    return x + (out.to(x.device) - x).detach()


def eot(x):
    choice = random.choice(["none", "noise", "resize", "jpeg", "noise", "resize"])
    if choice == "noise":
        x = x + torch.randn_like(x) * random.uniform(0.004, 0.012)
    elif choice == "resize":
        n = max(96, int(256 * random.uniform(0.82, 1.0)))
        x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    elif choice == "jpeg":
        x = jpeg_st(x, random.randint(68, 95))
    return quantize_st(x.clamp(0, 1))


def load_clean01():
    imgs = []
    for i in IDS:
        bgr = cv2.imread(f"{DATA}/clean_targets/{i}.png")
        imgs.append((bgr[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(imgs)).float().to(DEVICE)


def consensus_msg(verify_fn, dec, paths):
    B = np.stack([verify_fn(dec, cv2.imread(p)) for p in paths])
    return (B.mean(0) >= 0.5).astype(int), float((B == (B.mean(0) >= 0.5)).mean())


def save_forged(x01, outdir):
    os.makedirs(outdir, exist_ok=True)
    xn = x01.clamp(0, 1).detach().cpu().numpy()
    for k, i in enumerate(IDS):
        rgb = (xn[k].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        cv2.imwrite(f"{outdir}/{i}.png", rgb[:, :, ::-1])


def optimize(mode, m_R, m_P, R, P, steps, lr, lam, wP, marginR, marginP, rail, seed=0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    c = load_clean01(); B = c.shape[0]
    sR = (2 * torch.from_numpy(m_R.astype(np.float32)) - 1).unsqueeze(0).repeat(B, 1).to(DEVICE)
    sP = (2 * torch.from_numpy(m_P.astype(np.float32)) - 1).unsqueeze(0).repeat(B, 1).to(DEVICE)
    delta = torch.zeros_like(c, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    t0 = time.time()
    for s in range(steps):
        opt.zero_grad()
        x = eot((c + delta).clamp(0, 1))
        lR = logits_R(R, x)
        loss = torch.relu(marginR - sR * lR).mean()
        if mode == "ensemble":
            oP = logits_P(P, x)
            loss = loss + wP * torch.relu(marginP - sP * (oP - 0.5)).mean()
        pen = lam * delta.pow(2).sum() / B
        (loss + pen).backward(); opt.step()
        with torch.no_grad():
            delta.clamp_(-rail, rail)
        if s % 100 == 0 or s == steps - 1:
            print(f"  step {s:4d} loss={loss.item():.4f} pen={pen.item():.4f} dt={time.time()-t0:.0f}s", flush=True)
    return (c + delta).clamp(0, 1).detach(), c.detach()


def judge(outdir, R, P, m_R, m_P):
    """Held-out cross-decoder scorecard + LPIPS on the saved uint8 forgeries."""
    src = sorted(glob.glob(f"{DATA}/watermarked_sources/WM_3/*.png"))
    clean = [f"{DATA}/clean_targets/{i}.png" for i in IDS]
    # decoders' reading of REAL sources and their clean floor
    mR_src, _ = consensus_msg(verify_R, R, src)
    mP_src, _ = consensus_msg(verify_P, P, src)
    floorR = float((np.stack([verify_R(R, cv2.imread(p)) for p in clean]) ==
                    (np.stack([verify_R(R, cv2.imread(p)) for p in clean]).mean(0) >= 0.5)).mean())
    fp = sorted(glob.glob(f"{outdir}/*.png"))
    BR = np.stack([verify_R(R, cv2.imread(p)) for p in fp])
    BP = np.stack([verify_P(P, cv2.imread(p)) for p in fp])
    d = dict(
        R_to_msrc=float((BR == mR_src[None]).mean()),   # does forgery reproduce RoSteALS's source read
        P_to_msrc=float((BP == mP_src[None]).mean()),   # HELD-OUT unless ensemble
        R_to_mR=float((BR == m_R[None]).mean()),
        P_to_mP=float((BP == m_P[None]).mean()),
        R_floor=round(floorR, 3),
    )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ensemble", choices=["rosteals", "ensemble"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--wp", type=float, default=1.0)
    ap.add_argument("--marginR", type=float, default=2.0)
    ap.add_argument("--marginP", type=float, default=0.4)
    ap.add_argument("--rail", type=float, default=0.06)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    print(f"device={DEVICE} mode={a.mode} steps={a.steps} lam={a.lam} wp={a.wp} rail={a.rail}", flush=True)

    R, P = build_rosteals(), build_pimog()
    src = sorted(glob.glob(f"{DATA}/watermarked_sources/WM_3/*.png"))
    m_R, agR = consensus_msg(verify_R, R, src)
    m_P, agP = consensus_msg(verify_P, P, src)
    print(f"recovered m_R agree={agR:.3f}  m_P agree={agP:.3f}", flush=True)

    forged, clean = optimize(a.mode, m_R, m_P, R, P, a.steps, a.lr, a.lam, a.wp,
                             a.marginR, a.marginP, a.rail)
    save_forged(forged, a.outdir)
    lpips_l2 = float(((forged - clean) ** 2).mean().sqrt().item())
    card = judge(a.outdir, R, P, m_R, m_P)
    card.update(mode=a.mode, steps=a.steps, lam=a.lam, wp=a.wp, rail=a.rail,
                l2=round(lpips_l2, 4), m_R_agree=round(agR, 3), m_P_agree=round(agP, 3))
    with open(f"{a.outdir}/scorecard.json", "w") as f:
        json.dump(card, f, indent=2)
    print("SCORECARD " + json.dumps(card), flush=True)
    print(f">> HELD-OUT test: P_to_msrc={card['P_to_msrc']:.3f} vs R_floor~{card['R_floor']} "
          f"(ensemble uses PIMoG; for mode=rosteals this is the honest transfer signal)", flush=True)


if __name__ == "__main__":
    main()

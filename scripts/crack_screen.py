"""WM_5 crack screen for one candidate deep-watermark library.

Decodes the 25 same-message WM_5 sources with the candidate's decoder and reports
cross-image bit agreement + message balance, then embeds a message and reports the
residual's RGB channel-correlation signature. A genuine (verbatim-checkpoint) match
jumps to agreement >=0.95 with a balanced message AND reproduces WM_5's distinctive
*anti-correlated* channels (corr R,G=-0.39, G,B=-0.49). Anything landing in the
~0.6-0.92 band is same-family cross-talk (wrong weights) and will NOT transfer to the
real evaluator -- confirmed already for HiDDeN(0.925)/ARWGAN(0.84)/PIMoG(0.616)/FIN.

Run from inside the candidate's cloned repo so its modules import:
    cd SepMark && python3 ../crack_screen.py sepmark /path/to/Dataset [ckpt]
    cd CIN     && python3 ../crack_screen.py cin     /path/to/Dataset [ckpt]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

# We are run from inside the candidate's repo (cd SepMark && python3 .../crack_screen.py);
# Python only puts the script's own dir on sys.path, so add the cwd for `import network` etc.
sys.path.insert(0, os.getcwd())

WM5_TARGET = "-0.39, -0.49"  # WM_5's corr(R,G), corr(G,B)


def _cc(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _report(mode, agree, bal, resid):
    r = resid.reshape(-1, 3)
    rg, gb = _cc(r[:, 0], r[:, 1]), _cc(r[:, 1], r[:, 2])
    rms = float(np.sqrt((r ** 2).mean()))
    hit = agree >= 0.95 and 0.2 < bal < 0.8
    color_ok = rg < -0.2 and gb < -0.2
    print(f"[{mode}] WM_5 decode: agree={agree:.3f} bal={bal:.2f}  "
          f"residual corr(R,G)={rg:.3f} corr(G,B)={gb:.3f} rms={rms:.2f}")
    print(f"[{mode}] gate(agree>=0.95 & balanced)={'PASS' if hit else 'fail'}  "
          f"color-match(both<-0.2, WM_5={WM5_TARGET})={'PASS' if color_ok else 'fail'}")
    if hit and color_ok:
        print(f"[{mode}] *** CRACK CANDIDATE — re-embed and validate on the leaderboard ***")
    return hit and color_ok


def _load_tensors(data, size, rgb=True):
    srcs = sorted(glob.glob(f"{data}/watermarked_sources/WM_5/*.png"))
    tgts = sorted(glob.glob(f"{data}/clean_targets/*.png"),
                  key=lambda p: int(os.path.basename(p)[:-4]))
    tgts = [t for t in tgts if 101 <= int(os.path.basename(t)[:-4]) <= 125][:8]

    def to_t(p):
        im = Image.open(p).convert("RGB").resize((size, size))
        x = torch.from_numpy(np.asarray(im).transpose(2, 0, 1).astype(np.float32) / 255 * 2 - 1)
        return x.unsqueeze(0)

    return [to_t(s) for s in srcs], [to_t(t) for t in tgts]


def screen_sepmark(data, ckpt):
    from network.Decoder_U import DW_Decoder
    from network.Encoder_U import DW_Encoder

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sub = lambda p: {k[len(p) + 1:]: v for k, v in sd.items() if k.startswith(p + ".")}
    enc = DW_Encoder(30, attention="se").eval()
    dec = DW_Decoder(30, attention="se").eval()
    enc.load_state_dict(sub("encoder"), strict=True)
    dec.load_state_dict(sub("decoder_C"), strict=True)
    srcs, tgts = _load_tensors(data, 128)
    with torch.no_grad():
        M = np.stack([(dec(x) > 0).int().numpy()[0] for x in srcs])
        cons = M.mean(0)
        agree = float(np.maximum(cons, 1 - cons).mean())
        bal = float(cons.round().mean())
        msg = ((torch.tensor(cons.round()).float() * 2 - 1) * 0.1).unsqueeze(0)
        resid = np.concatenate(
            [((enc(x, msg) - x) / 2 * 255).squeeze().permute(1, 2, 0).numpy().reshape(-1, 3)
             for x in tgts], 0)
    return _report("sepmark", agree, bal, resid)


def screen_cin(data, ckpt):
    sys.path.insert(0, "codes")
    from models.CIN import CIN
    from utils.yml import dict_to_nonedict, parse_yml

    opt = dict_to_nonedict(parse_yml("codes/options/opt.yml"))
    opt["path"]["folder_temp"] = os.path.abspath("cin_tmp")
    os.makedirs("cin_tmp", exist_ok=True)
    opt["noise"]["Jpeg"] = dict(opt["noise"].get("Jpeg") or {})
    opt["noise"]["Jpeg"]["differentiable"] = False
    dev = "cpu"
    cin = CIN(opt, dev).to(dev).eval()
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["cinNet"]
    cin.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()},
                        strict=False)
    srcs, tgts = _load_tensors(data, 128)

    def decode(x):
        down = cin.invDown(x)
        cs = cin.cs_model(down, rev=True)
        inv = cin.invertible_model(cs, rev=True)
        _, m = cin.fusion_model(inv, None, cin.invDown, rev=True)
        return m.round().clamp(0, 1)

    with torch.no_grad():
        M = np.stack([decode(x).int().numpy()[0] for x in srcs])
        cons = M.mean(0)
        agree = float(np.maximum(cons, 1 - cons).mean())
        bal = float(cons.round().mean())
        msg = torch.tensor(cons.round()).float().unsqueeze(0)
        resid = np.concatenate(
            [((cin.encoder(x, msg) - x) / 2 * 255).squeeze().permute(1, 2, 0).numpy().reshape(-1, 3)
             for x in tgts], 0)
    return _report("cin", agree, bal, resid)


if __name__ == "__main__":
    mode = sys.argv[1]
    data = sys.argv[2]
    default_ckpt = {"sepmark": "EC_99.pth", "cin": "cin.pth"}[mode]
    ckpt = sys.argv[3] if len(sys.argv) > 3 else default_ckpt
    {"sepmark": screen_sepmark, "cin": screen_cin}[mode](data, ckpt)

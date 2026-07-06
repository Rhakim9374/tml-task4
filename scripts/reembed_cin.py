"""WM_5 = CIN (Combining Invertible & Non-invertible Mechanisms, ACM MM 2022).

Identified by the crack lottery: all 25 WM_5 sources decode to one balanced 30-bit
message with CIN's public checkpoint (agreement 1.000), and CIN's residual reproduces
WM_5's anti-correlated color signature. This re-embeds that message onto the WM_5
clean targets (101-125) with CIN's genuine encoder, so the evaluator's own decoder
reads it directly.

Run from inside the cloned CIN repo (github.com/rmpku/CIN), with cin.pth beside it:
    cd CIN && python3 /path/to/scripts/reembed_cin.py <Dataset> <out_dir> [cin.pth]
Deps: torch, torchvision, numpy, pillow, kornia, pyyaml. CPU is fine (128px).
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())   # CIN repo root (for `import ...` below)
sys.path.insert(0, "codes")


def load_cin(ckpt, dev="cpu"):
    from models.CIN import CIN
    from utils.yml import dict_to_nonedict, parse_yml

    opt = dict_to_nonedict(parse_yml("codes/options/opt.yml"))
    opt["path"]["folder_temp"] = os.path.abspath("cin_tmp")
    os.makedirs("cin_tmp", exist_ok=True)
    opt["noise"]["Jpeg"] = dict(opt["noise"].get("Jpeg") or {})
    opt["noise"]["Jpeg"]["differentiable"] = False
    cin = CIN(opt, dev).to(dev).eval()
    sd = torch.load(ckpt, map_location=dev, weights_only=False)["cinNet"]
    cin.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()},
                        strict=False)
    return cin


def to_t(path):
    im = Image.open(path).convert("RGB").resize((128, 128))
    return torch.from_numpy(np.asarray(im).transpose(2, 0, 1).astype(np.float32) / 255 * 2 - 1).unsqueeze(0)


def decode(cin, x):
    down = cin.invDown(x)
    cs = cin.cs_model(down, rev=True)
    inv = cin.invertible_model(cs, rev=True)
    _, m = cin.fusion_model(inv, None, cin.invDown, rev=True)
    return m.round().clamp(0, 1)


def main():
    data, out = sys.argv[1], sys.argv[2]
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "cin.pth"
    os.makedirs(out, exist_ok=True)
    cin = load_cin(ckpt)

    srcs = sorted(glob.glob(f"{data}/watermarked_sources/WM_5/*.png"))
    with torch.no_grad():
        M = np.stack([decode(cin, to_t(s)).int().numpy()[0] for s in srcs])
    cons = (M.mean(0) >= 0.5).astype(np.float32)
    print(f"consensus: agree={float((M == cons).mean()):.3f} bal={cons.mean():.2f} "
          f"msg={cons.astype(int).tolist()}")

    msg = torch.tensor(cons).unsqueeze(0)
    accs, rmss = [], []
    for n in range(101, 126):
        clean = to_t(f"{data}/clean_targets/{n}.png")
        with torch.no_grad():
            wm = cin.encoder(clean, msg)
            back = decode(cin, wm).int().numpy()[0]
        accs.append((back == cons.astype(int)).mean())
        rmss.append(float(((wm - clean) / 2 * 255).pow(2).mean().sqrt()))
        arr = ((wm.clamp(-1, 1)[0] / 2 + 0.5) * 255).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(f"{out}/{n}.png")
    print(f"re-embedded 25 -> {out}  round-trip BitAcc={np.mean(accs):.3f}  residual rms={np.mean(rmss):.2f} gl")


if __name__ == "__main__":
    main()

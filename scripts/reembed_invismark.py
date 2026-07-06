"""WM_3 = InvisMark (Microsoft, arXiv:2411.07795, 256px/100-bit, ~51 dB).

Identified by the crack lottery: the 25 WM_3 sources decode to one balanced 100-bit
message with InvisMark's public checkpoint (agreement 0.963 — the sub-1.0 is raw-bit
noise its ECC normally corrects). This re-embeds that message onto the WM_3 clean
targets (51-75) with InvisMark's genuine encoder.

Run from inside the cloned InvisMark repo (github.com/microsoft/InvisMark), with
paper.ckpt reachable:
    cd InvisMark && python3 /path/to/scripts/reembed_invismark.py <Dataset> <out_dir> <paper.ckpt>
Deps: torch, torchvision, timm, kornia, bchlib, pillow, numpy. CPU is fine (256px).
Loads only the Encoder/Extractor (the full Watermark class hardcodes .cuda()).
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.getcwd())  # InvisMark repo root (for `import model`, and to unpickle configs)


def main():
    import model

    data, out, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sz = tuple(cfg.image_shape) if isinstance(cfg.image_shape, (list, tuple)) else (cfg.image_shape,) * 2

    enc = model.Encoder(cfg)
    enc.load_state_dict(ck["encoder_state_dict"])
    enc.eval()
    dec = model.Extractor(cfg)
    dec.load_state_dict(ck["decoder_state_dict"])
    dec.eval()

    tf = transforms.Compose([transforms.Resize(sz), transforms.ToTensor(),
                             transforms.Normalize((0.5,) * 3, (0.5,) * 3)])

    def decode(x):
        with torch.no_grad():
            return (dec(x)[0] >= 0.5).int().numpy()

    srcs = sorted(glob.glob(f"{data}/watermarked_sources/WM_3/*.png"))
    M = np.stack([decode(tf(Image.open(p).convert("RGB")).unsqueeze(0)) for p in srcs])
    cons = (M.mean(0) >= 0.5).astype(np.float32)
    print(f"consensus: agree={float((M == cons).mean()):.3f} bal={cons.mean():.2f} "
          f"({cfg.num_encoded_bits}-bit)")

    secret = torch.tensor(cons).unsqueeze(0)
    accs, rmss = [], []
    for n in range(51, 76):
        clean = tf(Image.open(f"{data}/clean_targets/{n}.png").convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            wm = torch.clamp(enc(clean, secret), -1.0, 1.0)
            back = (dec(wm)[0] >= 0.5).int().numpy()
        accs.append((back == cons.astype(int)).mean())
        rmss.append(float(((wm - clean) / 2 * 255).pow(2).mean().sqrt()))
        arr = ((wm[0] / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(f"{out}/{n}.png")
    print(f"re-embedded 25 -> {out}  round-trip BitAcc={np.mean(accs):.3f}  residual rms={np.mean(rmss):.2f} gl")


if __name__ == "__main__":
    main()

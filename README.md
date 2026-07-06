# TML26 Task 4 — Watermark Forgery Attack

Steal the invisible watermark carried by a set of source images and imprint it onto
clean target images, so the targets are detected as carrying the watermark's message
while staying perceptually unchanged. Per-image score `S_final = S_det * S_qlt` with
`S_det = max(0, (BitAccuracy - 0.5) * 2)` and `S_qlt = exp(-8 * LPIPS(clean, forged))`.

**Setup.** 8 unknown watermarking methods `WM_1..WM_8`, each providing 25 images that
all carry the *same* hidden message. Fixed mapping onto the 200 clean targets: `WM_k`
→ targets `25(k-1)+1 .. 25k`. Resolutions: WM_5 is 128², WM_7/WM_8 are 512², the rest
256²; each group's sources share its target batch's resolution, so no resizing is needed.

## Approach: identify the scheme, then re-embed

The strongest attack is not to transfer a raw residual but to **recover the exact
watermarking algorithm**. Because a group's 25 sources share one message, we can decode
them with each candidate public watermark library and check whether all 25 yield a single
consistent, non-degenerate (balanced) message — confirmed by a re-embed round-trip. When a
library matches, we decode the shared message and re-embed it onto the clean targets with
that library's genuine encoder. This reproduces the real watermark, so the detector reads
it directly, at near-perfect bit-accuracy and imperceptible distortion — and it generalizes
to unseen images of the same scheme (no leaderboard tuning involved).

Groups with no matching public scheme fall back to a **denoiser-residual additive
transplant**: estimate the watermark as `mean_i(src_i − denoise(src_i))` (non-local-means
picks up the pattern best), then add `alpha · w_hat` to each target, scaling `alpha` per
image to a target LPIPS budget.

Identified schemes (each verified by a re-embed round-trip):

| Group | Targets | Scheme | Library |
|---|---|---|---|
| WM_1 | 1–25 | DwtDct (32-bit) | `invisible-watermark` |
| WM_2 | 26–50 | RivaGAN (32-bit) | `invisible-watermark` |
| WM_3 | 51–75 | *(additive transplant)* | — |
| WM_4 | 76–100 | VINE (VINE-R, 100-bit) | Shilin-LU/VINE |
| WM_5 | 101–125 | *(additive transplant)* | — |
| WM_6 | 126–150 | MBRS (MBRS_256, 256-bit) | jzyustc/MBRS |
| WM_7 | 151–175 | TrustMark model Q (ECC) | `trustmark` |
| WM_8 | 176–200 | TrustMark model P (raw) | `trustmark` |

Identification guards against degenerate false positives (raw decoders happily emit
all-0/all-1): a match requires high cross-image agreement **and** a balanced message
**and** a passing re-embed round-trip.

## Reproduce

```bash
git clone https://github.com/Rhakim9374/tml-task4.git && cd tml-task4
pip install -r requirements.txt      # base pipeline (numpy<2)
HF_REPO=SprintML/tml2026_task4 HF_TOKEN=hf_xxx bash cluster/fetch_data.sh
python -m scripts.extract_watermarks --dataset data/Dataset --out artifacts/wm.npz
```

The three neural schemes each run in their own environment and write forged PNGs into
`artifacts/<scheme>/` (setup commands are in the header of each script; all run on CPU or
GPU via `--device`):

```bash
python scripts/reembed_vine.py       --out artifacts/vine_wm4       # WM_4  (VINE env)
python scripts/reembed_mbrs.py       --out artifacts/mbrs_wm6       # WM_6  (MBRS env)
python scripts/reembed_trustmark.py  --out artifacts/trustmark_wm78 # WM_7/8 (TrustMark env, numpy>=2)
```

Assemble the submission — WM_1/WM_2 re-embed inline, WM_3/WM_5 transplant under the LPIPS
budget, and the neural forgeries merged in via `--prebuilt`:

```bash
python -m scripts.build_submission --watermarks artifacts/wm.npz --lpips-budget 0.02 \
    --prebuilt artifacts/vine_wm4 --prebuilt artifacts/mbrs_wm6 --prebuilt artifacts/trustmark_wm78 \
    --out submissions/best.zip
```

Submit (one per group per hour; the evaluation server is publicly reachable, so this can be
run from any machine with the zip):

```bash
export TML_API_KEY=<your key>
python -m scripts.submit --file submissions/best.zip
```

## Repository layout

```
src/
  data.py        dataset loader + fixed WM-group -> clean-target mapping
  denoise.py     denoiser zoo (median/gaussian/tv/wavelet/nlmeans/bilateral)
  extract.py     per-group watermark estimation + cross-half consistency scoring
  forge.py       additive imprint scaled to an LPIPS budget
  reembed.py     invisible-watermark (DwtDct/RivaGAN) scheme identification + re-embed
  quality.py     LPIPS quality score, S_det/S_qlt terms, PSNR
scripts/
  analyze.py             per-group forgeability diagnostic
  extract_watermarks.py  estimate + cache each group's additive watermark
  reembed_vine.py        VINE (WM_4) decode + re-embed          -> artifacts/vine_wm4
  reembed_mbrs.py        MBRS (WM_6) decode + re-embed           -> artifacts/mbrs_wm6
  reembed_trustmark.py   TrustMark (WM_7/8) decode + re-embed    -> artifacts/trustmark_wm78
  build_submission.py    assemble the 200-image zip (re-embed + transplant + --prebuilt)
  submit.py              POST the submission zip (key from TML_API_KEY)
cluster/
  fetch_data.sh    download + unpack the dataset
  pipeline.sub / run_pipeline.sh   extract + build a budget sweep (1 GPU)
  interactive.sub  interactive GPU job
```

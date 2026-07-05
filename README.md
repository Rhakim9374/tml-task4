# TML26 Task 4 — Watermark Forgery Attack

Steal the invisible watermark from a set of watermarked source images and imprint
it onto clean target images, so the target images are detected as carrying the
watermark's message — while staying perceptually unchanged.

Per-image score `S_final = S_det * S_qlt`, where
`S_det = max(0, (BitAccuracy - 0.5) * 2)` (0 at chance, 1 at perfect recovery) and
`S_qlt = exp(-8 * LPIPS(clean, forged))`. You must succeed at **both** strength and
quality: a strong watermark with a wrecked image scores ~0, and a pristine image
with no watermark scores ~0.

**Setup.** 8 unknown watermarking methods (`WM_1`..`WM_8`), each giving 25 images
that all carry the *same* hidden message. Fixed mapping onto the 200 clean targets:

| watermark | clean targets | resolution |
|-----------|---------------|------------|
| WM_1 | 1–25   | 256² |
| WM_2 | 26–50  | 256² |
| WM_3 | 51–75  | 256² |
| WM_4 | 76–100 | 256² |
| WM_5 | 101–125 | 128² |
| WM_6 | 126–150 | 256² |
| WM_7 | 151–175 | 512² |
| WM_8 | 176–200 | 512² |

Each group's sources share the resolution of its target batch, so no resizing is
needed when transplanting.

> **The README will be finalised with the exact best-result command once the
> leaderboard sweep settles.** The command below reproduces the current
> foundation submission (denoiser-residual transplant under an LPIPS budget).

## The approach in one paragraph

Each group's 25 sources share one hidden message, so for a content-agnostic
watermark the pattern is (roughly) a fixed additive signal. We recover it by
averaging denoiser residuals, `w_hat = mean_i(src_i - denoise(src_i))`, choosing
the denoiser per group by a cross-half consistency score. Non-local-means /
wavelet denoising strips self-similar image content while leaving the watermark,
which exposes it for **WM_3/4/5/6/8**. We then imprint `clean + alpha * w_hat`,
scaling `alpha` per image to a target **LPIPS budget** — the only quality signal
measurable without the hidden decoder. **WM_1/2/7** resist all denoisers (they are
content-adaptive / model-embedded) and are attacked with the heavier per-group
methods (learned denoisers, surrogate-decoder optimisation, WMCopier).

## Recreate it (foundation submission)

```bash
git clone https://github.com/Rhakim9374/tml-task4.git && cd tml-task4
pip install -r requirements.txt

# 1. Get the dataset into data/Dataset/  (the HF dataset is gated, so scp the zip)
scp data/Dataset.zip <user>@conduit2.hpc.uni-saarland.de:~/tml-task4/data/
bash cluster/fetch_data.sh                 # just extracts data/Dataset.zip

# 2. (diagnostic) how forgeable is each group?
python -m scripts.analyze --dataset data/Dataset

# 3. Extract each group's watermark estimate (auto denoiser per group)
python -m scripts.extract_watermarks --dataset data/Dataset --out artifacts/wm.npz

# 4. Build a submission zip at an LPIPS budget (or several with --budgets)
python -m scripts.build_submission --watermarks artifacts/wm.npz \
    --lpips-budget 0.04 --out submissions/best.zip

# 5. Submit (one per group per hour)
export TML_API_KEY=<your key>
python -m scripts.submit --file submissions/best.zip
```

On the HTCondor cluster, steps 3–4 run as one job: `condor_submit
cluster/pipeline.sub -append "args=--budgets 0.02,0.03,0.04,0.06"`.

## Repository layout

```
src/
  data.py        dataset loader + fixed WM-group -> clean-target mapping
  denoise.py     denoiser zoo (median/gaussian/tv/wavelet/nlmeans/bilateral)
  extract.py     per-group watermark estimation + cross-half consistency scoring
  forge.py       imprint watermark onto a target, scaled to an LPIPS budget
  quality.py     LPIPS quality score, S_det/S_qlt terms, PSNR sanity metric
scripts/
  analyze.py             CLI: per-group forgeability diagnostic
  extract_watermarks.py  CLI: estimate + cache each group's watermark (artifacts/wm.npz)
  build_submission.py    CLI: forge onto targets under an LPIPS budget -> zip
  submit.py              CLI: POST the submission zip (key from TML_API_KEY)
cluster/
  fetch_data.sh    extract (or download) the dataset into data/Dataset/
  pipeline.sub / run_pipeline.sh   extract + build a budget sweep (1 GPU)
  interactive.sub  interactive GPU job for the heavier (learned) methods
data/          dataset (gitignored)          plans/    internal notes (gitignored)
artifacts/     cached watermark estimates    submissions/ built zips (gitignored)
```

Provided course templates (`task_template.py`, `submission.py`) are kept locally
under `Reference/` (gitignored); our own code reimplements their logic in `src/`
and `scripts/`.

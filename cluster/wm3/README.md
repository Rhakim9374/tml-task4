# WM_3 forgery — parallel GPU sweep

WM_3 is the one unsolved group. Its watermark is RoSteALS-*family* (RoSteALS decoder reads it
+0.176 over its content floor; PIMoG +0.10) but not an exact match, so there is no clean crack.
This job forges WM_3 by **robust FNNS**: optimize each clean target so the RoSteALS decoder (solo)
or **RoSteALS + PIMoG jointly (ensemble)** read the recovered message, under EoT (jpeg/noise/resize)
so the mark is a real robust watermark, not a fragile adversarial one.

**Why the ensemble matters:** FNNS against RoSteALS alone produces a RoSteALS-*specific* mark — a
held-out decoder (PIMoG) reads it at chance, which predicts it won't transfer to the evaluator either.
Forcing BOTH decoders to read it makes the mark more universal. Each job's `scorecard.json` reports
the **held-out** decoder reading; a candidate is only worth submitting if a decoder NOT used to build
it reads it clearly above its content floor.

## Run (from the repo root on the cluster)

```bash
# 0. data present (once):  bash cluster/fetch_data.sh
# 1. one-time: if PIMoG's 5MB pretrained isn't in its repo, scp it from the laptop:
#    scp /private/tmp/PIMoG/models/ScreenShooting/Encoder_Decoder_Model_mask_99.pth \
#        <cluster>:<repo>/artifacts/wm3_models/pimog.pth
# 2. launch the parallel sweep (6 variants, 1 GPU each):
condor_submit cluster/wm3/wm3.sub
# 3. watch:  condor_q ;  tail -f runlogs/wm3_ens_base.out
```

Each variant writes `artifacts/wm3_out/<tag>/51.png..75.png` + `scorecard.json`.

## Pick the winner (held-out transfer)

```bash
for f in artifacts/wm3_out/*/scorecard.json; do echo "$f"; cat "$f"; echo; done
```

Rank by **`P_to_msrc`** (does the forgery reproduce what PIMoG reads on the REAL sources) for the
`ensemble` runs where PIMoG was *also* a target — but the honest signal is a decoder reading the mark
above its floor while NOT being the one it was optimized against. Prefer the highest held-out reading
at acceptable `l2` (lower = less distortion). If every variant's held-out reading sits at the floor,
FNNS does not transfer for WM_3 → keep the transplant baseline.

## Build + submit the winning candidate

```bash
python -m scripts.build_submission --dataset data/Dataset \
    --prebuilt artifacts/vine_wm4 --prebuilt artifacts/mbrs_wm6 \
    --prebuilt artifacts/trustmark_wm78 --prebuilt artifacts/cin_wm5 \
    --prebuilt artifacts/wm3_out/<winning_tag> \
    --out submissions/wm3_fnns.zip
python -m scripts.submit --file submissions/wm3_fnns.zip
```

Leaderboard is best-only, so a worse WM_3 cannot cost the standing 0.894760 — safe to test.

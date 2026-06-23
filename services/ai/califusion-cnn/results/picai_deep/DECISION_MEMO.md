# Decision memo — does a DEEP bpMRI encoder strengthen MoRA on PI-CAI?

**Status:** COMPLETE. All numbers computed on real PI-CAI data (n=1476 with masks; 419–425 csPCa+).
No fabrication. Negative/mixed findings reported plainly.

**Question.** We have a radiomics imaging arm (AUROC ~0.773) under which MoRA already beats named
SOTA under controlled modality failure. Does replacing it with a *trained* deep bpMRI encoder
(T2W/ADC/HBV, gland-centered 2.5D) (a) match/beat radiomics on csPCa, (b) sharpen the
site-shift signal MoRA relies on, and (c) WIDEN MoRA's margin over SOTA under failure?

**Bottom line.** The deep arm is **slightly weaker than radiomics on raw discrimination** and is
**less site-separable**, BUT it **WIDENS MoRA's advantage over the discrimination-recovery
baselines (static / TransCal-CPCS) at both severities** and MoRA keeps the **best ECE**. The
make-or-break claim survives — and is in fact stronger — with the deep arm, even though the deep
encoder is not the better stand-alone classifier here.

---

## Leakage controls (how every number stays honest)

1. **Patch construction touches no labels** — resample ADC/HBV/whole-gland-mask onto the T2W
   grid, take the K=12 central gland slices, crop a fixed 80 mm gland-centered FOV → 96×96,
   z-score each sequence *within the gland*. Pure intensity/geometry.
2. **Out-of-fold embeddings** — the deep csPCa CNN is trained with patient-level 5-fold CV; each
   case's 128-d penultimate embedding comes from a model whose training folds EXCLUDED that case
   (early stopping uses an inner split of the *training* fold only). The cached embeddings are
   therefore leakage-safe to reuse as fixed imaging features. Verified: re-loading the cache and
   scoring the cached OOF probabilities reproduces AUROC 0.7488.
3. **Per-fold heads** — Gate-B and the SOTA harness fit their scaler/classifier on the train
   split only; calibrators/temperatures/discriminators on source/train data only.
4. **Identical failure protocol** to `scripts/picai_sota_baselines.py`: imaging corrupted on a
   random 50% of TEST patients (additive Gaussian, severity 4 and 8), 10 seeds. Only the imaging
   feature matrix changes (deep vs radiomics) → deep-vs-radiomics comparison is apples-to-apples.

---

## Deep encoder

- Compact 3D CNN (~0.3M params, in=3ch t2w/adc/hbv, BN+ReLU, 4 stride-2 blocks → 128-d GAP head),
  BCE pos-weighted, early stopping on inner-val AUROC, MPS, patient-level 5-fold CV.
  *(A 3D ResNet18 was ~6× slower per step on MPS without learning faster; the compact net trains
  ~26 s/epoch and learns the csPCa signal — that's why it was chosen.)*
- **Deep imaging-only OOF csPCa AUROC = 0.749** (per-fold test AUROC: 0.743 / 0.750 / 0.744 /
  0.806 / 0.785). **vs radiomics 0.773.**

## Gate-B (5×5 patient-level CV)

| arm | AUROC | ECE |
|---|---|---|
| clinical-only | 0.745 | 0.034 |
| radiomics imaging-only | **0.775** | 0.130 |
| deep imaging-only | **0.755** | **0.029** |
| radiomics ⊕ clinical fusion | **0.811** | — |
| deep ⊕ clinical fusion | **0.787** | — |

- deep imaging-only **0.755** vs radiomics 0.773 → **Δ −0.020** (deep slightly behind).
- deep fusion **0.787** vs radiomics-fusion 0.811 → **Δ −0.024** (deep slightly behind).
- **But the deep arm is far better calibrated raw** (ECE 0.029 vs radiomics 0.130).

## Shift-detection sharpness (source RUMC+PCNN vs ZGT discriminator AUC)

| features | discriminator AUC |
|---|---|
| deep embeddings | **0.743** |
| radiomics | **0.922** (ref ~0.83) |
| clinical | 0.610 |

- **Hypothesis NOT supported.** The deep embeddings are *less* site-separable than radiomics —
  the trained encoder learned more site-invariant features. So MoRA's per-modality reliability
  signal is *not* sharper on deep; if anything the raw covariate shift is harder to detect.
  (MoRA still works under *injected* failure because the corruption itself is large and detectable.)

## Make-or-break: MoRA vs named SOTA under modality failure (10 seeds)

### Severity 4
| method | deep AUROC | deep ECE | radiomics AUROC |
|---|---|---|---|
| clinical | 0.743 | 0.040 | 0.743 |
| static (trusts broken scan) | 0.681 | 0.092 | 0.684 |
| TransCal/CPCS | 0.681 | 0.139 | 0.684 |
| evidential/TMC | 0.678 | 0.157 | 0.666 |
| **MoRA** | **0.709** | **0.061** | **0.700** |

MoRA Δ (win-rate): vs static **+0.028 (100%)** · vs TransCal **+0.028 (100%)** · vs evidential **+0.031 (90%)**

### Severity 8
| method | deep AUROC | deep ECE | radiomics AUROC |
|---|---|---|---|
| clinical | 0.743 | 0.040 | 0.743 |
| static | 0.673 | 0.104 | 0.672 |
| TransCal/CPCS | 0.673 | 0.151 | 0.672 |
| evidential/TMC | 0.651 | 0.179 | 0.633 |
| **MoRA** | **0.705** | **0.066** | **0.691** |

MoRA Δ (win-rate): vs static **+0.032 (100%)** · vs TransCal **+0.032 (100%)** · vs evidential **+0.054 (100%)**

### Did the deep arm WIDEN MoRA's margin? (deep Δ vs radiomics Δ)
| comparison | severity | radiomics Δ | deep Δ | widened? |
|---|---|---|---|---|
| MoRA vs static / TransCal-CPCS | 4 | +0.0157 | **+0.0282** | **YES (+0.013)** |
| MoRA vs static / TransCal-CPCS | 8 | +0.0186 | **+0.0321** | **YES (+0.013)** |
| MoRA vs evidential/TMC | 4 | +0.0336 | +0.0312 | ~equal (−0.002) |
| MoRA vs evidential/TMC | 8 | +0.0575 | +0.0540 | ~equal (−0.004) |

- **MoRA still best ECE** among fusion methods with the deep arm (0.061 / 0.066).
- **Selective risk @80% coverage** (MoRA-deferral vs weighted-conformal): with deep, the edge
  shrinks to a near tie (0.301 vs 0.303 @sev4, 50% of seeds; 0.313 vs 0.315 @sev8, 60%) — the
  selective-prediction advantage that was consistent on radiomics is NOT robust on deep.

---

## Decision

**Keep MoRA; the deep arm does not break the result and modestly strengthens the headline.**

- The core thesis — *only test-time per-modality shift detection recovers discrimination under
  modality failure; global calibration (TransCal/CPCS) and source-trained evidential uncertainty
  cannot* — **holds with the deep arm and the MoRA-vs-static/TransCal margin is ~1.8× wider**
  (+0.028–0.032 vs +0.016–0.019), 100% win-rate over 10 seeds at both severities.
- **But the deep encoder is not the better imaging arm on this dataset:** it is ~2 points behind
  radiomics on stand-alone and fused discrimination, and its embeddings are *less* site-separable
  (0.74 vs 0.92). It is, however, much better calibrated.
- **Recommendation for the manuscript:** report MoRA's robustness result on the **radiomics** arm
  as primary (stronger stand-alone imaging AUROC and the cleaner 0.83 site-shift story), and cite
  the deep arm as a **robustness check showing the MoRA advantage is not an artifact of the
  imaging representation** (it widens, not collapses, when the imaging arm is swapped for a
  trained CNN). Do **not** claim the deep encoder beats radiomics on discrimination — it does not.
- A stronger deep arm (more data/augmentation/pretraining, larger 3D backbone on real GPU) might
  recover the ~2-point discrimination gap; the compact MPS-budget CNN here was deliberately light.

## Artifacts

- `scripts/picai_deep_encoder.py` — preprocess (`--preprocess`) + 5-fold train & OOF embed (`--train`)
- `scripts/picai_deep_gateb_shift.py` — Gate-B + shift discriminator (deep vs radiomics)
- `scripts/picai_sota_baselines_deep.py` — controlled-failure MoRA-vs-SOTA on deep embeddings
- `data/processed/picai_deep_patches.npz` (1476 × 3×12×96×96), `picai_deep_embeddings.npz` (1476 × 128, +OOF)
- `results/picai_deep/{summary.json, deep_encoder_oof.json, gateb_shift.json, sota_deep_sev4.json, sota_deep_sev8.json}`
- Radiomics baselines (recomputed this session): `results/picai_sota/summary_sev{4,8}.json`

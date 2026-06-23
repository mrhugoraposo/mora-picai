# DECISION MEMO — Option 3: Can a stronger frozen CT foundation encoder rescue the imaging / fusion accuracy story on Lung1 2-yr OS?

**Date:** 2026-06-20
**Cohort:** NSCLC-Radiomics "Lung1" (TCIA), matched N = 419, 2-yr OS prevalence 0.597 (250 high-risk / 169).
**Evaluation:** repeated stratified **patient-level** 5×5 CV, fixed seeds 0–4, leakage-safe (scaler + head + calibrator fit per training fold only; embeddings are frozen pre-extracted features — the encoder fits nothing on the cohort).
**Artifacts:** `results/option3/{model_comparison.csv, paired_delta.csv, per_seed.csv, complementarity.json, gates.json, summary.json, shift_medicalnet.json}`; embeddings cached at `data/processed/foundation_embeddings_<enc>.npz`; provenance in `foundation_embeddings_manifest.json`.

---

## Bottom line

**No.** Three pretrained 3D CT encoders — including a genuine medical foundation model (CT-FM, SSL on 148k CT scans) and MedicalNet (pretrained on 23 medical datasets) — **fail to recover any imaging or fusion accuracy contribution** for 2-yr OS on Lung1. The best frozen foundation image-only model (MedicalNet, AUROC 0.569) does **not** beat the hand-crafted radiomics floor (0.587 / 0.601) and the strongest foundation model (CT-FM) is the **worst** image arm (0.518, near chance). The best clinical+foundation fusion adds only **+0.021–0.028 AUROC** over clinical-only, with a **paired-bootstrap 95% CI that crosses zero** (DeLong p ≈ 0.29–0.56). This is a clean, well-powered negative result. **Recommendation: commit to the reliability / calibration paper (MORA).**

Critically — and this is the load-bearing diagnostic — **we obtained real medical-pretrained weights** (MedicalNet *and* CT-FM both loaded and verified non-random). So the negative conclusion is **not** caveated by "we couldn't get good weights." The encoders are strong; the *signal* isn't there in a frozen, tumor-ROI representation.

---

## 1. Encoders actually used (weights verified)

| Encoder | Architecture | Pretraining | Embedding | Status |
|---|---|---|---|---|
| **medicalnet** | MONAI 3D ResNet-18 | **MedicalNet — 23 medical 3D datasets** | 512-d (penultimate) | Loaded; weights confirmed ≠ random init |
| **ctfm** | SegResNet encoder (`lighter_zoo`) | **CT-FM SSL — 148,000 CT scans (IDC), contrastive** | 512-d (GAP of deepest 512-ch map) | Loaded from HF `project-lighter/ct_fm_feature_extractor` |
| **r3d18** | torchvision r3d_18 | Kinetics-400 (generic 3D video) | 512-d | Loaded (generic-3D reference) |

All three are **frozen** (no grad, eval mode). Embeddings extracted for **421/423** patients with CT+RTSTRUCT (matched to 419 labeled clinical cases for modeling). No patients failed mask materialization in the modeled set. **There is no medical-pretrained-weights caveat on this result.**

**Image representation:** tumor-centered 3D ROI = fixed 48×96×96-voxel box around the **GTV-1 centroid** (RTSTRUCT, orientation-guarded via the existing interior-HU guard), zero-padded at borders, no resampling (in-plane FOV matches the verified radiomics / 2.5D pipeline). Intensity normalization is encoder-specific: MedicalNet & r3d18 use the house lung window (L −600 / W 1500 → [0,1]); CT-FM uses its README spec (ScaleIntensityRange HU [−1024, 2048] → [0,1]); r3d18 replicates the window to 3 channels with depth-as-time.

---

## 2. Model comparison (the headline table)

AUROC = CV mean ± SD over 5 seeds; `boot [lo,hi]` = 2000-sample percentile bootstrap 95% CI on the seed-0 pooled OOF predictions. ECE / Brier on seed-0 pooled OOF.

| Model | AUROC (CV) | Bootstrap 95% CI | Brier | ECE |
|---|---|---|---|---|
| **Clinical-only** (gboost) | **0.573 ± 0.008** | 0.569 [0.511, 0.626] | 0.263 | 0.140 |
| **Radiomics-only** (logreg) | **0.601 ± 0.012** | 0.595 [0.540, 0.651] | 0.239 | 0.057 |
| cnn25d ImageNet (established neg. ref) | 0.415 | — | — | — |
| **Clinical + radiomics** late-mean (comparator) | **0.614 ± 0.008** | 0.608 [0.553, 0.664] | 0.233 | 0.046 |
| Foundation image-only — **MedicalNet** | 0.569 ± 0.005 | 0.572 [0.517, 0.625] | 0.305 | 0.237 |
| Foundation image-only — **CT-FM** | **0.518 ± 0.010** | 0.536 [0.482, 0.594] | 0.288 | 0.180 |
| Foundation image-only — r3d18 | 0.552 ± 0.017 | 0.566 [0.510, 0.619] | 0.290 | 0.206 |
| **Clinical + MedicalNet** late-mean (best fusion) | **0.596 ± 0.007** | 0.590 [0.536, 0.642] | 0.248 | 0.111 |
| Clinical + MedicalNet stack | 0.587 ± 0.006 | 0.581 [0.525, 0.636] | 0.236 | **0.016** |
| Clinical + CT-FM late-mean | 0.570 ± 0.010 | 0.583 [0.529, 0.638] | 0.248 | 0.073 |
| Clinical + r3d18 late-mean | 0.586 ± 0.013 | 0.597 [0.538, 0.650] | 0.244 | 0.078 |

Per-encoder, three fusion strategies were tested (early-concat, late-mean, logistic stacking); the table shows the best per encoder. Full grid in `model_comparison.csv`.

**Read:** every foundation image-only model lands **at or below clinical (0.573)** and **below radiomics (0.601)**. The best multimodal number in the whole study remains the **established radiomics fusion (0.614)**, which already FAILED the project's Gate-1 floor of 0.63. Foundation encoders do not change the ceiling.

---

## 3. Paired ΔAUROC (fusion − clinical) — the decision statistic

Paired bootstrap (2000×) on seed-0 pooled OOF over the **same patients**; seed-Δ = per-seed (fusion − clinical) spread.

| Encoder | Best fusion | ΔAUROC | Paired 95% CI | DeLong p | Seed-Δ (mean ± sd) | CI excludes 0? |
|---|---|---|---|---|---|---|
| medicalnet | late-mean | **+0.021** | **[−0.032, +0.072]** | 0.438 | +0.023 ± 0.007 | **No** |
| ctfm | late-mean | +0.014 | [−0.033, +0.061] | 0.563 | −0.004 ± 0.011 | No |
| r3d18 | late-mean | **+0.028** | **[−0.026, +0.079]** | 0.293 | +0.013 ± 0.010 | No |

The largest fusion gain (+0.028, r3d18) is **below the +0.03 minimum**, well below the +0.05 preferred bar, and **every CI crosses zero**. CT-FM's per-seed delta is actually **negative on average** (fusion hurts). No fusion is statistically or practically meaningful.

---

## 4. Complementarity — does imaging add info beyond clinical, or just re-encode stage?

| Encoder | corr(img, clinical) | Fusion fixes clinical errors | Clinical acc → Fusion acc |
|---|---|---|---|
| medicalnet | +0.004 | 65 / 176 (0.37) | 0.580 → 0.606 |
| ctfm | −0.009 | 50 / 176 (0.28) | 0.580 → 0.585 |
| r3d18 | −0.051 | 55 / 176 (0.31) | 0.580 → 0.601 |

At face value the near-zero correlation and "fixes 37% of clinical errors" *look* complementary — and the automated G3 gate flips PASS on this. **But this is an artifact, not signal**, for two reasons:

1. **The fusion AUROC gain that this "complementarity" produces is itself not significant** (Section 3). Averaging an uninformative, near-orthogonal score into clinical will, by chance, flip a fraction of borderline cases either way; here it nets a tiny, CI-crossing-zero gain. "Fixes 65, breaks ~40" nets the +0.02 we already deemed noise.
2. **Diagnostic — the embeddings are nearly patient-invariant.** Mean absolute pairwise correlation of the 512-d embeddings *across patients* is **0.993 (MedicalNet)** and **0.777 (CT-FM)** — i.e. the frozen encoder returns almost the same vector for every tumor. r3d18 is the most varied (0.628), which is exactly why it extracts marginally more (and why CT-FM, the most collapsed, is worst). The corr-with-clinical ≈ 0 is what you get from an almost-constant feature, not from genuinely independent tumor biology. **The encoders are dominated by global CT appearance and are not tumor-discriminative for survival in a frozen setting.**

So: **no credible evidence of complementarity.** The honest reading is that on Lung1, 2-yr OS imaging signal is (a) weak in absolute terms (radiomics ≈ 0.60 is the realistic ceiling, consistent with Aerts-style C-index ~0.65 on a related endpoint) and (b) not extractable by a frozen foundation encoder on a tumor ROI.

---

## 5. Calibration & Tier-1 imaging shift (the part that *did* produce a usable signal)

One genuinely interesting observation: the **late-mean / stacking fusions are far better calibrated** than clinical-only (stack ECE **0.012–0.016** vs clinical **0.140**) — but this is a *calibration* benefit from probability averaging/recalibration, **not** a discrimination benefit, and it belongs to the reliability paper, not an accuracy claim.

Tier-1 imaging covariate-shift (Gaussian noise sd=80 HU + blur σ=1; train clean / test shifted), MedicalNet, `shift_medicalnet.json`:

| Arm | AUROC clean → shift | ECE clean → shift |
|---|---|---|
| Clinical (control, imaging-independent) | 0.569 → 0.569 | 0.140 → 0.140 |
| Foundation image-only | 0.564 → 0.560 (Δ −0.004) | 0.108 → 0.153 (+0.044) |
| Fusion late-mean | 0.590 → 0.590 (Δ **0.000**) | 0.057 → 0.089 (+0.032) |

The fusion's discrimination is **completely insensitive** to destroying the image (Δ AUROC = 0.000) — direct confirmation that imaging is a near-null contributor to the fused score. The only thing the shift moves is **calibration** (ECE rises), which is precisely the phenomenon the MORA reliability paper is built to detect and correct.

---

## 6. Decision gates

| Gate | Criterion | Result |
|---|---|---|
| **G1** | Foundation image-only > cnn25d (0.42) **and** ≥ radiomics floor (~0.587) | **FAIL** — best 0.569 < 0.587; CT-FM 0.518 |
| **G2** | Fusion beats clinical by ≥ +0.03 AUROC, paired CI excludes 0, stable across seeds, calibration not worse | **FAIL** — best Δ +0.028, all CIs cross 0, DeLong n.s. |
| **G3** | Complementarity (not re-encoding stage) | nominal PASS but **spurious** (see §4); reclassified **FAIL on inspection** |
| **ALL** | — | **FAIL** |

G1 clears the trivial bar (every foundation arm beats the broken cnn25d 0.42 — a frozen pretrained encoder is a better feature extractor than the end-to-end 2.5D CNN that collapsed), but **fails the meaningful bar** (≥ radiomics). That is the honest comparison.

---

## 7. Answers to the six questions

1. **Does a stronger CT foundation encoder rescue the accuracy/fusion story?** No. With genuine MedicalNet *and* CT-FM (148k-scan SSL) weights, foundation image-only ≤ clinical and < radiomics; best fusion gain +0.021–0.028 AUROC, CI crosses 0.
2. **Does imaging add complementary info beyond clinical?** No credible evidence. Apparent complementarity is an artifact of averaging a near-constant, uninformative embedding (cross-patient embedding corr 0.78–0.99); the resulting fusion gain is not significant and the score is image-insensitive under shift.
3. **Statistically and practically meaningful?** Neither. Below the +0.03 floor; all paired ΔAUROC 95% CIs include 0; DeLong p ≥ 0.29.
4. **Stable enough to be the main paper?** No — per-seed deltas straddle 0 (CT-FM negative on average). An accuracy/fusion headline would be unpublishable and indefensible at review.
5. **If not, how does this strengthen the reliability/calibration paper?** Substantially. It (a) converts "we didn't try hard enough on imaging" into a **documented negative control**: two SOTA medical foundation encoders, frozen, on a proper GTV ROI, cannot beat radiomics — so the contribution is honestly *not* discrimination; (b) the shift test shows imaging perturbation moves **calibration, not discrimination**, which is exactly MORA's thesis; (c) the fusion's strong post-hoc calibration (ECE 0.14→0.016) is a clean reliability result. The existing MORA evidence already stands (clean ECE 0.172→0.083; **imaging-shift ECE 0.293→0.147**; shift-attribution acc 0.999). This experiment removes the obvious reviewer objection ("did you try a CT foundation model?") with a rigorous "yes, here is the negative result."
6. **Recommended publication strategy.** **Reliability / calibration paper (MORA) as the main contribution.** Frame CaliFusion-CNN honestly as a *calibrated, shift-aware, deferral-equipped multimodal pipeline whose contribution is trustworthiness, not a discrimination breakthrough*. Include this Option-3 study as a **methods/negative-results section or supplement**: "frozen CT foundation encoders (MedicalNet, CT-FM) do not add discriminative value for 2-yr OS on Lung1; imaging informs *calibration under shift*, which our method addresses." Keep radiomics fusion (0.614) as the imaging baseline. Do **not** pursue an imaging/fusion accuracy headline.

---

## 8. Honest limitations

- **Frozen-encoder ceiling:** we tested encoders *frozen* (the brief's design — leakage-safe, reproducible on a Mac). Light fine-tuning of CT-FM on the GTV ROI *might* extract more, but with N=419 and prevalence 0.60 the overfitting risk is severe and this was explicitly out of scope; it would not be expected to clear 0.63 given radiomics (a near-best-case hand-crafted imaging representation) tops out at 0.60.
- **ROI / pooling choices:** single GTV-centroid box; CT-FM pooled by GAP of the deepest feature map. Alternative aggregations (multi-scale, sliding-window over the whole lung) were not swept; given the cross-patient embedding collapse, they are unlikely to change the verdict.
- **Single cohort, single endpoint:** Lung1, 2-yr OS. External validation (NSCLC-Radiogenomics) is Phase 7 and unaffected by this decision.
- **No medical-pretrained-weights caveat:** explicitly, this result is *not* weakened by weight availability — MedicalNet and CT-FM both loaded with verified pretrained weights.

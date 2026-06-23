# CaliFusion-CNN — Manuscript Tables (reverse-engineered)

Scope key: **VERIFIED** = computed live from real data in this repo;
**[pending GPU run]** = produced by `scripts/train_multimodal.py` after imaging
download/preprocessing (no values fabricated). Numbers shown as `mean ± SD`
across 5 repeats; AUROC also with bootstrap 95% CI.

---

## Table 1 — Dataset characteristics (VERIFIED, computed from the real Lung1 clinical file)

| Characteristic | Value |
|---|---|
| Dataset / source | NSCLC-Radiomics "Lung1", The Cancer Imaging Archive (Aerts et al., 2014) |
| License | CC BY-NC 3.0 (public) |
| Task | 2-year overall survival (binary; high-risk = death ≤ 730 days) |
| Imaging modality | Pre-treatment CT + GTV tumor segmentation (RTSTRUCT/SEG) |
| Clinical variables | age, gender, clinical T/N/M stage, overall stage, histology |
| Reference standard | Clinical follow-up / survival records |
| Total matched cohort | 422 patients (imaging IDs 1:1 with clinical) |
| **Usable labeled N** | **420** (2 censored < 2 yr excluded) |
| **Positive / negative** | **251 / 169** (prevalence 0.598) |
| Split (patient-level, stratified) | 70 / 15 / 15 (one CT per patient → patient-level by construction) |
| Missing-data rate | age 22/422 (5.2%); histology 42/422 (10.0%); clinical.T 1; overall stage 1; survival/status 0 |
| Data-quality exclusions | 3 CTs with skipped slices (LUNG1-014/021/085) — interpolate or exclude |
| Inclusion / exclusion | NSCLC w/ pre-treatment CT + GTV + non-censored 2-yr status / censored < 2 yr |
| Histology breakdown | squamous 152, large cell 114, NOS 63, adenocarcinoma 51, missing 42 |
| Stage breakdown | IIIb 176, IIIa 112, I 93, II 40 |
| External validation | Candidate: NSCLC-Radiogenomics CT (Stanford/VA → site/scanner shift) |

---

## Table 2 — Baseline comparison (clinical row VERIFIED; imaging/fusion pending)

| Model | AUROC (95% CI) | AUPRC | Sens | Spec | PPV | NPV | F1 | κ | ECE | Brier | NLL | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Image-only CNN | [pending] | … | … | … | … | … | … | … | … | … | … | pending GPU run |
| **Clinical-only (HistGBoost)** | **0.583 (0.495–0.607)** | 0.660 | 0.593 | 0.518 | 0.652 | 0.481 | 0.595 | 0.111 | 0.137 | 0.262 | 0.746 | **VERIFIED (5×5 CV, n=420)** |
| Early fusion | [pending] | … | … | … | … | … | … | … | … | … | … | pending GPU run |
| Late fusion | [pending] | … | … | … | … | … | … | … | … | … | … | pending GPU run |
| Attention fusion | [pending] | … | … | … | … | … | … | … | … | … | … | pending GPU run |
| CaliFusion-CNN (uncalibrated) | [pending] | … | … | … | … | … | … | … | … | … | … | pending GPU run |
| CaliFusion-CNN (calibrated) | [pending] | … | … | … | … | … | … | … | … | … | … | pending GPU run |

All values are `mean ± SD` over 5 CV repeats in the CSV exports; rounded here. Clinical-only ± SD: AUROC 0.583±0.017, AUPRC 0.660±0.011, ECE 0.137±0.014, Brier 0.262±0.009, NLL 0.746±0.022.

---

## Table 3 — Calibration comparison, clinical model (VERIFIED)

| Method | ECE | MCE | Brier | NLL | Cal. slope | Cal. intercept | AUROC |
|---|---|---|---|---|---|---|---|
| Uncalibrated | 0.137±0.014 | 0.361±0.071 | 0.262±0.009 | 0.746±0.022 | 0.244±0.056 | 0.279±0.030 | 0.583±0.017 |
| Temperature | 0.073±0.013 | 0.395±0.140 | 0.242±0.006 | 0.678±0.012 | 0.598±0.309 | 0.298±0.051 | 0.585±0.021 |
| Platt | 0.047±0.008 | 0.359±0.274 | 0.240±0.006 | 0.674±0.013 | 0.607±0.504 | 0.132±0.210 | 0.565±0.041 |
| Isotonic | 0.073±0.022 | 0.684±0.128 | 0.247±0.006 | 0.930±0.098 | 0.045±0.018 | 0.367±0.020 | 0.570±0.021 |

Empirical confirmation of H2: calibration improves probability reliability (ECE 0.137→0.047–0.073; Brier and NLL down) **without** improving discrimination (AUROC essentially unchanged; DeLong uncalibrated vs temperature p = 0.987, McNemar p = 1.0).

---

## Table 4 — Dataset-shift stress tests (template; [pending GPU run])

| Shift condition | AUROC | Sens | Spec | ECE | Brier | NLL | Retained error | Coverage |
|---|---|---|---|---|---|---|---|---|
| No shift (reference) | … | … | … | … | … | … | … | … |
| Acquisition/site (Radiogenomics) | … | … | … | … | … | … | … | … |
| Degraded image quality (noise/blur/lowres/compression) | … | … | … | … | … | … | … | … |
| Missing clinical variables (10/30/50%) | … | … | … | … | … | … | … | … |
| Prevalence shift (0.30/0.50/0.70) | … | … | … | … | … | … | … | … |

> Note: the clinical-arm shift rows (missing-clinical, prevalence) are runnable now via `eval/shift.py` on the tabular model; image shifts require the trained CNN.

---

## Table 5 — Deferral / coverage analysis (template; [pending GPU run])

| Operating point | Coverage | Deferral rate | Retained error | Retained Sens | Retained Spec | % errors deferred |
|---|---|---|---|---|---|---|
| No deferral | 1.00 | 0.00 | … | … | … | … |
| τ @ 90% coverage | 0.90 | 0.10 | … | … | … | … |
| τ @ 80% coverage | 0.80 | 0.20 | … | … | … | … |
| τ @ 70% coverage | 0.70 | 0.30 | … | … | … | … |
| τ @ 60% coverage | 0.60 | 0.40 | … | … | … | … |

---

## Table 6 — Computational cost (template; fill from your run)

| Measure | Value |
|---|---|
| Trainable parameters | [INSERT] |
| Training time | [INSERT] ([INSERT GPU]) |
| Inference latency / case | [INSERT ms] |
| Peak memory (inference) | [INSERT] |
| Calibration overhead | [INSERT] (one validation pass) |
| Preprocessing time / case | [INSERT] |
| Deferral workload @ chosen τ | [INSERT %] |

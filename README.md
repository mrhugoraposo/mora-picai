# CaliFusion-CNN / MoRA

**Modality-Aware Reliability (MoRA): test-time, per-modality reliability for safe
multimodal clinical decision support — analysis code and research console.**

A label-free, test-time reliability estimator that detects when one input modality
goes out-of-distribution at inference, reweights multimodal fusion toward the
trustworthy modality, and abstains with modality-attributed conformal risk control.
The method is validated for clinically-significant prostate cancer (csPCa) detection
on the public **PI-CAI** cohort (biparametric MRI + structured clinical variables);
the repository also contains the **NSCLC-Radiomics ("Lung1")** imaging-redundancy and
probability-calibration audit, and an authenticated web console for running and
visualizing the experiments.

> **Data boundary.** This repository ships **code, configuration, and computed result
> metrics only** — **no patient imaging, no marksheet, no derived embeddings**. All
> datasets must be obtained from their official sources (below). The console mounts
> any local models/data **read-only**; they are never committed or baked into images.

## Repository contents

| Path | What |
|---|---|
| `services/ai/califusion-cnn/` | Analysis pipeline — reproduces the paper's tables/figures (clinical baseline, PI-CAI MoRA vs SOTA, Lung1 audit). |
| `services/api/` | **MoRA Research Console** — FastAPI web app: reliability-gated inference + experiment / back-testing views (the running app). |
| `apps/web/reviewer/` | Static reviewer companion (single page). |
| `infrastructure/docker/` | Containerized deployment for the console (Compose + Postgres). |

## Datasets (obtain separately)

| Dataset | Use | License | Source |
|---|---|---|---|
| **PI-CAI** | csPCa detection (MoRA) | CC BY-NC | Imaging: Zenodo [10.5281/zenodo.6624726](https://doi.org/10.5281/zenodo.6624726). Labels/masks: `DIAGNijmegen/picai_labels` (marksheet + delineations). |
| **NSCLC-Radiomics ("Lung1")** | imaging-redundancy / calibration audit | CC BY-NC 3.0 | The Cancer Imaging Archive (TCIA). |

Place raw data under `services/ai/califusion-cnn/data/raw/` and derived features under
`…/data/processed/` (git-ignored). See `services/ai/califusion-cnn/data/README_DATA.md`.

## Reproduce the analysis

```bash
cd services/ai/califusion-cnn
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt

python scripts/run_clinical_baseline.py            # clinical/tabular baseline (CPU)
python scripts/picai_radiomics.py                  # PI-CAI MRI radiomics
python scripts/picai_fusion.py                     # radiomics + clinical fusion
python scripts/picai_sota_baselines.py             # MoRA vs static / TransCal-CPCS / weighted-conformal / evidential
python scripts/picai_deep_encoder.py               # deep bpMRI encoder (robustness)
# Lung1 redundancy & calibration audit:
python scripts/radiomics_diagnostic.py scripts/fusion_radiomics.py scripts/lung1_hostmarkers.py
```

Patient-level splits, fixed seeds `[0–4]`, bootstrap CIs throughout; computed metric
files are provided under `services/ai/califusion-cnn/results/` for verification.

## Run the research console

```bash
# Local (SQLite, no Docker) — reuses the pipeline environment above
cd services/api && ./run_local.sh 8080             # http://127.0.0.1:8080  (admin / changeme by default)

# Containerized (Postgres)
cd infrastructure/docker && cp .env.example .env    # set strong SECRET_KEY / *_PASSWORD
docker compose up -d
```

### Try it with synthetic data (no setup, no patient data)

The console ships a self-contained synthetic demonstration — no PI-CAI data required:

```bash
cd services/api
python build_demo.py        # generate non-identifiable synthetic demo models
./run_local.sh 8080         # http://127.0.0.1:8080  (admin / changeme)
```

Sign in, pick a `DEMO-###` case, and toggle **break imaging** to watch the imaging
reliability collapse and MoRA down-weight imaging (or defer). All demo data is randomly
generated — no patient data is involved.

**Security.** The console enforces login but is **not hardened for the open
internet** — bind it to LAN/loopback and put it behind a reverse proxy with HTTPS or
a VPN; never expose the port directly. Inference is **CPU-only** (no GPU/torch at
runtime), using persisted MoRA reliability references.

## Reproducibility
- Patient-level splits (no leakage); calibrators and decision thresholds fit on
  validation folds only.
- Fixed seeds `[0,1,2,3,4]`; percentile bootstrap 95% CIs; DeLong / McNemar / Holm.
- Config-driven; no hard-coded local paths; no fabricated numbers — every reported
  value traces to a real run, with artifacts under `results/`.

## Citation
If you use this software, please cite the archived release (see `CITATION.cff` and the
repository's Zenodo record), and the datasets you use: **PI-CAI** —
Zenodo [10.5281/zenodo.6624726](https://doi.org/10.5281/zenodo.6624726) (and the PI-CAI
challenge / Saha et al., 2024); **NSCLC-Radiomics** — Aerts et al. (2014) and the TCIA
collection.

## License
Code: **MIT** (see `LICENSE`). Datasets are licensed separately (CC BY-NC) and are
**not** redistributed here.

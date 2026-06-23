# Data acquisition & licensing

This repository does **not** redistribute patient data (no imaging, no marksheet,
no derived embeddings). Obtain each dataset from its official source and place it
under `data/raw/` (git-ignored).

## PI-CAI — biparametric prostate MRI + clinical (primary cohort, MoRA)
- Imaging: **Zenodo 10.5281/zenodo.6624726** (public release; ~27 GB).
- Labels & masks: **`DIAGNijmegen/picai_labels`** (GitHub) — the clinical / AI
  marksheet and whole-gland + lesion delineations.
- **License: CC BY-NC** (non-commercial; attribution required). See the Zenodo
  record for the exact license version.
- Primary citation: the PI-CAI challenge / Saha A. et al., *The Lancet Oncology* 2024.
- Target layout:
  ```
  data/raw/PI-CAI/<patient>/<study>/{t2w,adc,hbv}.mha
  data/raw/PI-CAI/marksheet.csv     # operator-supplied; git-ignored, never commit
  data/processed/                   # derived features / caches written by the pipeline
  ```

## NSCLC-Radiomics ("Lung1") — CT + clinical (imaging-redundancy / calibration audit)
- Collection: **NSCLC-Radiomics** ("Lung1"), The Cancer Imaging Archive (TCIA).
- Modalities: pre-treatment **CT** + **GTV tumor segmentation** (RTSTRUCT/SEG).
- Clinical CSV (survival, stage, histology, age, gender), version 3 (Oct 2019);
  the URL is in `configs/default.yaml` and is fetched automatically by
  `scripts/run_clinical_baseline.py`.
- **License: CC BY-NC 3.0** (non-commercial; attribution required).
- Primary citation: Aerts HJWL et al., *Nature Communications* 2014; plus the TCIA
  collection DOI and the TCIA data-usage citation.
- How to download imaging:
  ```bash
  # Option A — scripted NBIA REST API (edit DOWNLOAD=True first; ≈30+ GB)
  python scripts/download_tcia_lung1.py
  # Option B — NBIA Data Retriever (GUI/CLI), or: pip install nbiatoolkit
  ```
- Target layout: `data/raw/NSCLC-Radiomics/<PatientID>/<SeriesInstanceUID>/*.dcm`

## Ethics / governance
Both collections are public and de-identified; non-commercial use only. Do not
attempt re-identification. Keep all raw data and derived patient-level artifacts
out of version control (see the repository `.gitignore`).

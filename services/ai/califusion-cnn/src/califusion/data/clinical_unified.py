"""
califusion.data.clinical_unified  —  cross-cohort clinical schema for the fusion arm.

RESEARCH_DESIGN §7 directs the multimodal **clinical encoder** to use the Lung1 ∩
NSCLC-Radiogenomics field intersection so the same encoder can be evaluated externally
(Tier 2, qualitative). The intersection is deliberately leaner than the verified
clinical-only anchor (which uses full clinical T/N/M + Overall stage and scores AUROC
0.583); the reduced signal is expected and documented.

Unified schema (4 features):
  numeric      : age
  categorical  : gender, overall_stage (coarse I/II/III/IV), histology (coarse 4-way)

Cohort mappings
  Lung1 (NSCLC-Radiomics, clinical v3):
    age           <- "age"
    gender        <- "gender"            (male/female)
    overall_stage <- coarse("Overall.Stage")   I, II, IIIa, IIIb, IV -> I/II/III/IV
    histology     <- coarse("Histology")        adeno / squamous / large cell / nos
  Radiogenomics (Phase G — NOT materialized here):
    age           <- "Age at Histological Diagnosis"
    gender        <- "Gender"
    overall_stage <- DERIVED from pathological T/N/M (or AJCC if present).  ** NOT 1:1 **:
                     clinical (Lung1) vs pathological (Radiogenomics) staging differ and
                     ~23% are "Not Collected". Resolve + document at Phase G; Tier 2 is
                     qualitative-only (Gate 2 resolved), so this does not gate Phase B/C.
    histology     <- "Histology " (trailing space)

Leakage: this module only *derives raw features* (no fitting). The encoder/preprocessor
(`build_unified_preprocessor`) is fit on TRAIN folds only, inside the training loop.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .clinical_preprocess import load_clinical, build_labels, CLINICAL_CSV_URL, ID_COL

UNIFIED_NUMERIC = ["age"]
UNIFIED_CATEGORICAL = ["gender", "overall_stage", "histology"]
UNIFIED_FEATURES = UNIFIED_NUMERIC + UNIFIED_CATEGORICAL


def _coarse_stage_lung1(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower().replace("stage", "").strip()
    if s.startswith("iv") or s == "4":
        return "IV"
    if s.startswith("iii") or s in ("3", "3a", "3b") or s in ("iiia", "iiib"):
        return "III"
    if s.startswith("ii") or s == "2":
        return "II"
    if s.startswith("i") or s == "1":
        return "I"
    return np.nan


def _coarse_histology(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    if "adeno" in s:
        return "adenocarcinoma"
    if "squamous" in s:
        return "squamous"
    if "large" in s:
        return "large_cell"
    if "nos" in s or "not otherwise" in s:
        return "nos"
    return "other"


def _gender(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    if s.startswith("m"):
        return "male"
    if s.startswith("f"):
        return "female"
    return np.nan


def lung1_to_unified(usable_df: pd.DataFrame) -> pd.DataFrame:
    """Map a Lung1 'usable' dataframe (from build_labels) to the unified 4-feature schema."""
    out = pd.DataFrame(index=usable_df.index)
    out["age"] = pd.to_numeric(usable_df.get("age"), errors="coerce")
    out["gender"] = usable_df.get("gender").map(_gender)
    out["overall_stage"] = usable_df.get("Overall.Stage").map(_coarse_stage_lung1)
    out["histology"] = usable_df.get("Histology").map(_coarse_histology)
    return out[UNIFIED_FEATURES]


def build_unified_preprocessor() -> ColumnTransformer:
    """Leakage-safe transformer (fit on TRAIN folds only) for the unified schema."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, UNIFIED_NUMERIC),
        ("cat", cat_pipe, UNIFIED_CATEGORICAL),
    ])


def get_unified_xy_lung1(path_or_url: str = CLINICAL_CSV_URL):
    """Return (usable_df, X_unified_raw_df, y, patient_ids) for Lung1 on the unified schema.

    Same labelling/exclusions as the verified clinical baseline; only the feature columns
    differ (intersection schema). Raw features (NaN preserved) — transform per-split later.
    """
    df = load_clinical(path_or_url)
    df = df.replace({"NA": np.nan, "Na": np.nan, "na": np.nan, "": np.nan})
    usable, y = build_labels(df)
    X = lung1_to_unified(usable)
    ids = usable[ID_COL].to_numpy()
    return usable, X, y, ids

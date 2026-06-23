"""
califusion.data.picai_clinical — PI-CAI clinical arm (csPCa detection).

Loads the public PI-CAI marksheet (DIAGNijmegen/picai_labels), derives the binary
csPCa label (ISUP≥2), and exposes the leakage-safe clinical feature schema. Verified
(Gate-0, ADR-0009): clinical-only AUROC 0.741 (5×5 CV) / 0.788 train(RUMC+PCNN)→test(ZGT).

Pre-MRI clinical features ONLY (no post-hoc/biopsy leakage):
  numeric : patient_age, psa, psad, prostate_volume
`center` ∈ {RUMC, PCNN, ZGT} drives patient-level splits and the REAL vendor/site shift
(train RUMC+PCNN → test ZGT = Siemens→Philips, 80 positives). `patient_id` aligns to imaging.

Leakage: features are raw (NaN preserved); preprocessor fit per training split.
"""
from __future__ import annotations
import io
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MARKSHEET_URL = ("https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/"
                 "main/clinical_information/marksheet.csv")
NUMERIC = ["patient_age", "psa", "psad", "prostate_volume"]
ID_COL = "patient_id"
CENTER_COL = "center"
_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "PI-CAI", "marksheet.csv")


def load_marksheet(path_or_url: str = None) -> pd.DataFrame:
    """Load the marksheet, caching it locally on first download (small CSV)."""
    if path_or_url and os.path.exists(path_or_url):
        return pd.read_csv(path_or_url)
    if os.path.exists(_CACHE):
        return pd.read_csv(_CACHE)
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(MARKSHEET_URL, headers={"User-Agent": "califusion/1.0"})
    raw = urllib.request.urlopen(req, timeout=60, context=ctx).read().decode("utf-8-sig", "replace")
    df = pd.read_csv(io.StringIO(raw))
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    df.to_csv(_CACHE, index=False)
    return df


def build_labels(df: pd.DataFrame):
    """Binary csPCa (ISUP≥2) from case_csPCa (YES/NO)."""
    y = (df["case_csPCa"].astype(str).str.upper() == "YES").astype(int).to_numpy()
    return y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
    ])


def get_picai_clinical(path_or_url: str = None):
    """Return (df, X_raw[NUMERIC], y, patient_ids, center)."""
    df = load_marksheet(path_or_url)
    y = build_labels(df)
    X = df[NUMERIC].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return df, X, y, df[ID_COL].to_numpy(), df[CENTER_COL].astype(str).to_numpy()

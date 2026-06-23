"""
califusion.data.clinical_preprocess
Lung1 (NSCLC-Radiomics) clinical data loading, 2-year overall-survival labelling,
and a leakage-safe preprocessing pipeline (impute + encode + scale) reused by the
tabular baseline and the multimodal clinical encoder.

Label (binary high 2-yr mortality risk):
    positive (1) = deadstatus.event == 1 AND Survival.time <= HORIZON_DAYS
    negative (0) = Survival.time > HORIZON_DAYS
    excluded     = censored before HORIZON (deadstatus.event == 0 AND Survival.time <= HORIZON)
"""
from __future__ import annotations
import io
import urllib.request
import ssl
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CLINICAL_CSV_URL = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
)
HORIZON_DAYS = 730
NUMERIC = ["age"]
CATEGORICAL = ["gender", "clinical.T.Stage", "Clinical.N.Stage",
               "Clinical.M.Stage", "Overall.Stage", "Histology"]
ID_COL = "PatientID"


def load_clinical(path_or_url: str = CLINICAL_CSV_URL) -> pd.DataFrame:
    if path_or_url.startswith("http"):
        ctx = ssl.create_default_context(); ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "Mozilla/5.0 research"})
        raw = urllib.request.urlopen(req, timeout=60, context=ctx).read().decode("utf-8-sig", "replace")
        return pd.read_csv(io.StringIO(raw))
    return pd.read_csv(path_or_url)


def build_labels(df: pd.DataFrame, horizon: int = HORIZON_DAYS):
    """Return (df_usable, y) after applying the 2-yr OS labelling and exclusions."""
    df = df.copy()
    df["Survival.time"] = pd.to_numeric(df["Survival.time"], errors="coerce")
    df["deadstatus.event"] = pd.to_numeric(df["deadstatus.event"], errors="coerce")
    surv = df["Survival.time"]; dead = df["deadstatus.event"]
    pos = (dead == 1) & (surv <= horizon)
    neg = surv > horizon
    keep = pos | neg
    out = df.loc[keep].copy()
    y = pos.loc[keep].astype(int).to_numpy()
    return out.reset_index(drop=True), y


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
    ])


def get_xy(path_or_url: str = CLINICAL_CSV_URL):
    """Convenience: returns (df_usable, X_raw_df, y, patient_ids)."""
    df = load_clinical(path_or_url)
    # normalise NA tokens to real NaN
    df = df.replace({"NA": np.nan, "Na": np.nan, "na": np.nan, "": np.nan})
    usable, y = build_labels(df)
    X = usable[NUMERIC + CATEGORICAL].copy()
    # numeric stays float w/ NaN; categoricals -> str w/ NaN preserved (stage cols
    # are numeric-with-NaN and would otherwise mix float+str after imputation)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in CATEGORICAL:
        X[c] = X[c].map(lambda v: np.nan if (v is None or (isinstance(v, float) and np.isnan(v)))
                        else str(v).strip())
    ids = usable[ID_COL].to_numpy()
    return usable, X, y, ids

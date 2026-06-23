#!/usr/bin/env python3
"""
scripts/build_deployment_models.py — persist the MoRA deployment models for the web app.

Trains and saves (joblib) the artifacts the live FastAPI service needs so that inference
on a NEW case never re-reads patient imaging and never re-fits on test data. Everything is
fit on the SOURCE distribution only (leakage-safe), exactly mirroring the verified scripts:
  - scripts/picai_sota_baselines.py  (PI-CAI MoRA: disc_reliability r=clip((1-d)/(1-d_ref),0,1)^λ)
  - scripts/picai_fusion.py          (PI-CAI Gate-B fusion numbers)
  - scripts/radiomics_diagnostic.py  (Lung1 radiomics arm)
  - scripts/run_clinical_baseline.py (Lung1 clinical arm)

For each dataset (PI-CAI, Lung1) we persist:
  * imaging model    : LogisticRegression on the radiomics features (source-fit)
  * clinical model   : LogisticRegression on the clinical numeric features (source-fit)
  * per-modality preprocessors (median impute + standard scale), fit on SOURCE
  * MoRA reliability components per modality:
        - the SOURCE feature reference matrix X_src (scaled) — so a NEW case's reliability
          can be computed by training a fresh domain discriminator src-vs-{the new point(s)};
        - d_ref (median in-distribution discriminator level on source);
        - lambda (sharpness exponent).
  * an operating threshold (Youden-J on the source OOF predictions of the fused score).

We also store the SOURCE label vector so the discriminator's d_ref and the operating
threshold are reproducible at inference time without any target labels.

Outputs:
  models/picai_imaging.joblib, models/picai_clinical.joblib, models/picai_reliability.joblib
  models/lung1_imaging.joblib, models/lung1_clinical.joblib, models/lung1_reliability.joblib
  models/manifest.json    (artifact registry + the verified metrics each arm reproduces)

Run:  python scripts/build_deployment_models.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from califusion.data.picai_clinical import load_marksheet, NUMERIC as PICAI_CLIN

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
EPS = 1e-6
LAMBDA = 2.0  # MoRA reliability sharpness, matches picai_sota_baselines.py default


# ----------------------------------------------------------------------------- helpers
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def ece(y, p, nb=12):
    b = np.linspace(0, 1, nb + 1)
    e = 0.0
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def fit_scaler(X):
    """Median-impute + standard-scale, fit on source. Returns fitted Pipeline."""
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler())]).fit(X)


def reliability_components(X_src_scaled, lam=LAMBDA):
    """Compute d_ref = median in-distribution discriminator level on the SOURCE.

    At inference, reliability of a NEW (scaled) point x_t is:
        train LR discriminator on [X_src_scaled (dom 0); x_t (dom 1)] (5-fold OOF on source side
        is not needed for a single point — we fit on source vs the target batch and read d_t),
        r = clip((1 - d_t) / (1 - d_ref), 0, 1) ** lam.
    Here we only need to persist X_src_scaled, d_ref, lam. d_ref is estimated by a source-vs-source
    null: split source in half and read the in-distribution discriminator level (≈0.5 by symmetry,
    but we store the empirical median to stay faithful to disc_reliability)."""
    n = len(X_src_scaled)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    half = n // 2
    dom = np.zeros(n)
    dom[perm[:half]] = 1.0  # arbitrary half labelled "target" to read the in-distribution level
    d = cross_val_predict(
        Pipeline([("s", StandardScaler()),
                  ("c", LogisticRegression(max_iter=2000, C=0.5))]),
        X_src_scaled, dom, cv=5, method="predict_proba")[:, 1]
    d_ref = float(np.median(d))
    return {"X_src_scaled": X_src_scaled.astype(np.float32), "d_ref": d_ref, "lambda": lam}


# ----------------------------------------------------------------------------- PI-CAI
def build_picai():
    rad = pd.read_csv(os.path.join(PROC, "picai_radiomics_features.csv"))
    mk = load_marksheet()
    mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    df = rad.merge(mk[["patient_id", "label", "center"] + PICAI_CLIN],
                   on="patient_id", how="inner").reset_index(drop=True)
    rad_cols = [c for c in rad.columns
                if c not in ("patient_id", "study_id") and not c.startswith("Unnamed")]
    rad_cols = [c for c in rad_cols if df[c].notna().any()]
    y = df["label"].to_numpy(int)

    Xi_raw = df[rad_cols].apply(pd.to_numeric, errors="coerce")
    Xc_raw = df[PICAI_CLIN].apply(pd.to_numeric, errors="coerce")

    # source preprocessors (fit on all available = the source pool RUMC+PCNN+ZGT)
    si = fit_scaler(Xi_raw)
    sc = fit_scaler(Xc_raw)
    Xi = si.transform(Xi_raw)
    Xc = sc.transform(Xc_raw)

    # per-modality models
    mi = LogisticRegression(max_iter=2000, C=0.5).fit(Xi, y)
    mc = LogisticRegression(max_iter=2000, C=0.5).fit(Xc, y)

    # honest in-sample + OOF AUROC for the manifest (OOF = leakage-free estimate)
    oof_i = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Xi, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    oof_c = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Xc, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    p_img_full = mi.predict_proba(Xi)[:, 1]
    p_clin_full = mc.predict_proba(Xc)[:, 1]
    fused_oof = 0.5 * (oof_i + oof_c)  # static fuse for the operating threshold
    thr = youden_threshold(y, fused_oof)

    rel_img = reliability_components(Xi)
    rel_clin = reliability_components(Xc)

    metrics = {
        "imaging_auroc_oof": round(float(roc_auc_score(y, oof_i)), 4),
        "clinical_auroc_oof": round(float(roc_auc_score(y, oof_c)), 4),
        "static_fusion_auroc_oof": round(float(roc_auc_score(y, fused_oof)), 4),
        "imaging_ece_oof": round(ece(y, oof_i), 4),
        "clinical_ece_oof": round(ece(y, oof_c), 4),
        "n": int(len(df)), "positives": int(y.sum()),
        "operating_threshold": round(thr, 4),
        "centers": {k: int(v) for k, v in df["center"].value_counts().to_dict().items()},
        "note": "Verified powered Gate-B (picai_fusion.py): fusion 0.804 vs clinical 0.741, "
                "Δ+0.063. MoRA recovers under modality failure (picai_sota_baselines.py).",
    }

    joblib.dump({"model": mi, "scaler": si, "feature_cols": rad_cols,
                 "modality": "imaging", "dataset": "picai"},
                os.path.join(MODELS, "picai_imaging.joblib"))
    joblib.dump({"model": mc, "scaler": sc, "feature_cols": PICAI_CLIN,
                 "modality": "clinical", "dataset": "picai"},
                os.path.join(MODELS, "picai_clinical.joblib"))
    joblib.dump({"imaging": rel_img, "clinical": rel_clin,
                 "operating_threshold": thr, "lambda": LAMBDA,
                 "imaging_feature_cols": rad_cols, "clinical_feature_cols": PICAI_CLIN,
                 "y_source": y.astype(np.int8)},
                os.path.join(MODELS, "picai_reliability.joblib"))

    # persist a small replay table (ids + raw features) so the app can REPLAY cases
    replay = df[["patient_id", "center", "label"] + PICAI_CLIN + rad_cols].copy()
    replay.to_csv(os.path.join(MODELS, "picai_replay.csv"), index=False)

    print(f"[PI-CAI] n={len(df)} pos={y.sum()} | imaging OOF {metrics['imaging_auroc_oof']} "
          f"clinical OOF {metrics['clinical_auroc_oof']} fused {metrics['static_fusion_auroc_oof']} "
          f"| thr {thr:.3f} | feats img={len(rad_cols)} clin={len(PICAI_CLIN)}")
    return metrics, rad_cols


# ----------------------------------------------------------------------------- Lung1
def build_lung1():
    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    rad_cols = [c for c in rad_cols if rad[c].notna().any()]
    y = rad["label"].to_numpy(int)
    Xi_raw = rad[rad_cols].apply(pd.to_numeric, errors="coerce")

    # clinical arm — unified clinical schema (age + gender/stage/histology one-hot)
    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    # align rows by PatientID
    merged = rad[["PatientID"]].merge(clin, on="PatientID", how="left")
    clin_num = ["age"]
    clin_cat = ["gender", "overall_stage", "histology"]
    Xc_raw = merged[clin_num + clin_cat].copy()

    si = fit_scaler(Xi_raw)
    Xi = si.transform(Xi_raw)

    # clinical preprocessor: impute+scale numeric, one-hot categorical
    clin_prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), clin_num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), clin_cat),
    ]).fit(Xc_raw)
    Xc = clin_prep.transform(Xc_raw)
    if hasattr(Xc, "toarray"):
        Xc = Xc.toarray()

    mi = LogisticRegression(max_iter=2000, C=0.5).fit(Xi, y)
    mc = LogisticRegression(max_iter=2000, C=1.0).fit(Xc, y)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof_i = cross_val_predict(LogisticRegression(max_iter=2000, C=0.5), Xi, y, cv=cv,
                              method="predict_proba")[:, 1]
    oof_c = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Xc, y, cv=cv,
                              method="predict_proba")[:, 1]
    fused_oof = 0.5 * (oof_i + oof_c)
    thr = youden_threshold(y, fused_oof)

    rel_img = reliability_components(Xi)
    rel_clin = reliability_components(Xc)

    metrics = {
        "imaging_auroc_oof": round(float(roc_auc_score(y, oof_i)), 4),
        "clinical_auroc_oof": round(float(roc_auc_score(y, oof_c)), 4),
        "static_fusion_auroc_oof": round(float(roc_auc_score(y, fused_oof)), 4),
        "n": int(len(rad)), "positives": int(y.sum()),
        "operating_threshold": round(thr, 4),
        "note": "Lung1 = supporting negative control: 4 imaging approaches add NO complementary "
                "discrimination over clinical (~0.58). Radiomics floor ~0.60, clinical ~0.58. "
                "Redundancy strengthens the reliability thesis (see results/option3).",
    }

    joblib.dump({"model": mi, "scaler": si, "feature_cols": rad_cols,
                 "modality": "imaging", "dataset": "lung1"},
                os.path.join(MODELS, "lung1_imaging.joblib"))
    joblib.dump({"model": mc, "preprocessor": clin_prep,
                 "numeric": clin_num, "categorical": clin_cat,
                 "modality": "clinical", "dataset": "lung1"},
                os.path.join(MODELS, "lung1_clinical.joblib"))
    joblib.dump({"imaging": rel_img, "clinical": rel_clin,
                 "operating_threshold": thr, "lambda": LAMBDA,
                 "imaging_feature_cols": rad_cols,
                 "y_source": y.astype(np.int8)},
                os.path.join(MODELS, "lung1_reliability.joblib"))

    replay = rad[["PatientID", "label"] + rad_cols].merge(
        merged[["PatientID"] + clin_num + clin_cat], on="PatientID", how="left")
    replay.to_csv(os.path.join(MODELS, "lung1_replay.csv"), index=False)

    print(f"[Lung1]  n={len(rad)} pos={y.sum()} | imaging OOF {metrics['imaging_auroc_oof']} "
          f"clinical OOF {metrics['clinical_auroc_oof']} fused {metrics['static_fusion_auroc_oof']} "
          f"| thr {thr:.3f} | feats img={len(rad_cols)}")
    return metrics, rad_cols


def main():
    os.makedirs(MODELS, exist_ok=True)
    print("Building deployment models -> models/\n")
    picai_m, picai_feats = build_picai()
    lung1_m, lung1_feats = build_lung1()

    manifest = {
        "schema_version": 1,
        "built_by": "scripts/build_deployment_models.py",
        "mora_reliability": {
            "formula": "r = clip((1 - d_t) / (1 - d_ref), 0, 1) ** lambda",
            "lambda": LAMBDA,
            "discriminator": "LogisticRegression(C=0.5) domain classifier source-vs-target",
            "source": "PI-CAI: RUMC+PCNN+ZGT pool; Lung1: full matched cohort. d_ref = median "
                      "in-distribution discriminator level on a source-vs-source null split.",
            "fusion": "p_mora = (r_img*p_img + r_clin*p_clin) / (r_img + r_clin)",
        },
        "leakage_controls": "All scalers, models, reliability references fit on SOURCE only. "
                            "Inference never uses target labels; reliability uses source-fit "
                            "reference + a fresh source-vs-newcase discriminator.",
        "artifacts": {
            "picai_imaging": "models/picai_imaging.joblib",
            "picai_clinical": "models/picai_clinical.joblib",
            "picai_reliability": "models/picai_reliability.joblib",
            "picai_replay": "models/picai_replay.csv",
            "lung1_imaging": "models/lung1_imaging.joblib",
            "lung1_clinical": "models/lung1_clinical.joblib",
            "lung1_reliability": "models/lung1_reliability.joblib",
            "lung1_replay": "models/lung1_replay.csv",
        },
        "datasets": {
            "picai": {**picai_m, "task": "csPCa detection (ISUP>=2) on prostate bpMRI",
                      "imaging_features": len(picai_feats),
                      "clinical_features": PICAI_CLIN},
            "lung1": {**lung1_m, "task": "2-year overall survival (NSCLC-Radiomics Lung1)",
                      "imaging_features": len(lung1_feats),
                      "clinical_features": ["age", "gender", "overall_stage", "histology"]},
        },
    }
    with open(os.path.join(MODELS, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote models/manifest.json + {len(manifest['artifacts'])} artifacts")


if __name__ == "__main__":
    main()

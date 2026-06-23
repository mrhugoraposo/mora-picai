#!/usr/bin/env python3
"""
scripts/fusion_radiomics.py  —  Gate 1 re-test with radiomics as the imaging modality.

The deep 2.5D CNN imaging arm failed Gate 1 (AUROC ~0.42, encoder bottleneck). This tests
whether a *working* imaging representation — hand-crafted GTV radiomics (first-order +
shape + gradient + LoG + GLCM) — fused with the unified clinical features clears the
~0.63 signal floor. Tabular + leakage-safe (per-fold fit), 5x5 patient-level CV.

Fusion strategies:
  early_concat : concat radiomics+clinical features -> single model
  late_mean    : mean of per-modality calibrated probabilities
Per-modality best model chosen from {logreg, gboost}.

Outputs results/fusion_radiomics/summary.json + a Gate 1 verdict.
Run:  python scripts/fusion_radiomics.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "fusion_radiomics")
GATE1 = 0.63
REPEATS, FOLDS = 5, 5


def _pre(num, cat):
    t = [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num)]
    if cat:
        t.append(("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="UNK")),
                                   ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(t)


def _model(kind, num, cat):
    clf = (LogisticRegression(max_iter=2000, C=0.5) if kind == "logreg"
           else HistGradientBoostingClassifier(random_state=0))
    return Pipeline([("p", _pre(num, cat)), ("c", clf)])


def oof_probs(df, y, num, cat, kind, seed):
    X = df; oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = clone(_model(kind, num, cat)); m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def cv_auroc(df, y, num, cat, kind):
    a = [roc_auc_score(y, oof_probs(df, y, num, cat, kind, s)) for s in range(REPEATS)]
    return float(np.mean(a)), float(np.std(a))


def main():
    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))
    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    df = rad.merge(clin.drop(columns=["label"]), on="PatientID", how="inner")
    y = df["label"].to_numpy(int)
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    cnum, ccat = ["age"], ["gender", "overall_stage", "histology"]
    print(f"merged n={len(df)} | radiomics={len(rad_cols)} | prev={y.mean():.3f}")

    res = {}
    # single modalities (best model each)
    for name, (num, cat) in {"clinical": (cnum, ccat), "radiomics": (rad_cols, [])}.items():
        best = max((("logreg", cv_auroc(df, y, num, cat, "logreg")),
                    ("gboost", cv_auroc(df, y, num, cat, "gboost"))), key=lambda t: t[1][0])
        res[name] = {"model": best[0], "auroc": round(best[1][0], 4), "sd": round(best[1][1], 4)}

    # early fusion (concat)
    ef = max((("logreg", cv_auroc(df, y, rad_cols + cnum, ccat, "logreg")),
              ("gboost", cv_auroc(df, y, rad_cols + cnum, ccat, "gboost"))), key=lambda t: t[1][0])
    res["fusion_early"] = {"model": ef[0], "auroc": round(ef[1][0], 4), "sd": round(ef[1][1], 4)}

    # late fusion (mean of per-modality probs, per seed) using each modality's best model
    rk, ck = res["radiomics"]["model"], res["clinical"]["model"]
    la = []
    for s in range(REPEATS):
        pr = oof_probs(df, y, rad_cols, [], rk, s)
        pc = oof_probs(df, y, cnum, ccat, ck, s)
        la.append(roc_auc_score(y, 0.5 * (pr + pc)))
    res["fusion_late_mean"] = {"model": f"{rk}+{ck}", "auroc": round(float(np.mean(la)), 4),
                               "sd": round(float(np.std(la)), 4)}

    best_fusion = max(res["fusion_early"]["auroc"], res["fusion_late_mean"]["auroc"])
    passed = best_fusion >= GATE1
    summary = {"n": len(df), "cv": f"{REPEATS}x{FOLDS}", "gate1_threshold": GATE1,
               "results": res, "best_fusion_auroc": best_fusion,
               "deep_cnn_image_auroc": 0.415,
               "gate1_decision": "PASS" if passed else "FAIL",
               "note": ("radiomics fusion clears the floor — viable multimodal path"
                        if passed else
                        "best multimodal ~%.3f < 0.63; modalities largely redundant on Lung1 2-yr OS"
                        % best_fusion)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Gate 1 re-test (radiomics imaging modality) ===")
    for k, v in res.items():
        print(f"  {k:18s} {v['model']:14s} AUROC {v['auroc']:.3f} ± {v['sd']:.3f}")
    bar = "=" * 60
    print(f"\n{bar}\nGATE 1: best multimodal AUROC = {best_fusion:.3f} (threshold {GATE1})")
    print(f"  vs deep-CNN image arm 0.415 | DECISION: {summary['gate1_decision']}")
    print(f"  {summary['note']}\n{bar}")


if __name__ == "__main__":
    main()

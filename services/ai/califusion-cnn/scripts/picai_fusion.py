#!/usr/bin/env python3
"""
scripts/picai_fusion.py — PI-CAI Gate-B: does MRI radiomics ⊕ clinical add value? (ADR-0009)

Merges prostate-MRI radiomics (picai_radiomics_features.csv) with the clinical arm
(picai_clinical: age/psa/psad/volume, AUROC ~0.74) on patient_id, and compares
clinical-only / radiomics-only / fusion (early-concat, late-mean) with 5×5 patient-level
CV AUROC + ECE. If all three centers are present with adequate counts, also runs the REAL
cross-site shift: train RUMC+PCNN → test ZGT (Gate-C preview).

Leakage-safe: preprocessors fit per fold; cross-site uses source centers only for fit.
Works on whatever cases are extracted (fold0 subset for a first read; all folds later).

Run:  python scripts/picai_fusion.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from califusion.data.picai_clinical import load_marksheet, NUMERIC as CLIN

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
REP, FOLDS = 5, 5


def ece(y, p, nb=15):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def pipe(kind):
    clf = LogisticRegression(max_iter=2000, C=0.5) if kind == "logreg" else HistGradientBoostingClassifier(random_state=0)
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", clf)])


def cv_oof(X, y, kind, seed):
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = clone(pipe(kind)); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def cv(X, y, kind):
    a = []
    for s in range(REP):
        oof = cv_oof(X, y, kind, s); a.append(roc_auc_score(y, oof))
    oofN = cv_oof(X, y, kind, 0)
    return float(np.mean(a)), float(np.std(a)), ece(y, oofN)


def main():
    rad = pd.read_csv(os.path.join(PROC, "picai_radiomics_features.csv"))
    mk = load_marksheet()
    mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    df = rad.merge(mk[["patient_id", "label", "center"] + CLIN], on="patient_id", how="inner")
    y = df["label"].to_numpy(int)
    rad_cols = [c for c in rad.columns if c not in ("patient_id", "study_id")]
    print(f"merged n={len(df)} | csPCa+={y.sum()} ({y.mean():.3f}) | radiomics={len(rad_cols)} feats")
    print("per-center:", df.assign(y=y).groupby("center")["y"].agg(["size", "sum"]).to_dict("index"))

    Xc = df[CLIN].to_numpy(float); Xr = df[rad_cols].to_numpy(float); Xf = np.hstack([Xc, Xr])
    print("\n=== Gate-B: 5×5 CV AUROC (±sd) | ECE ===")
    res = {}
    for name, X in [("clinical", Xc), ("radiomics", Xr), ("fusion_early", Xf)]:
        best = max((("logreg", cv(X, y, "logreg")), ("gboost", cv(X, y, "gboost"))), key=lambda t: t[1][0])
        res[name] = best
        print(f"  {name:14s} {best[0]:7s} AUROC {best[1][0]:.3f} ± {best[1][1]:.3f} | ECE {best[1][2]:.3f}")
    # late fusion
    ck, rk = res["clinical"][0], res["radiomics"][0]
    la = []
    for s in range(REP):
        pc = cv_oof(Xc, y, ck, s); pr = cv_oof(Xr, y, rk, s); la.append(roc_auc_score(y, 0.5 * (pc + pr)))
    print(f"  {'fusion_late':14s} {ck}+{rk:7s} AUROC {np.mean(la):.3f} ± {np.std(la):.3f}")
    best_fusion = max(res["fusion_early"][1][0], float(np.mean(la)))
    dlt = best_fusion - res["clinical"][1][0]
    print(f"\n  ΔAUROC fusion − clinical = {dlt:+.3f} | clinical {res['clinical'][1][0]:.3f} "
          f"radiomics {res['radiomics'][1][0]:.3f} fusion {best_fusion:.3f}")
    print(f"  Gate-B (fusion > clinical, complementary imaging): "
          f"{'PASS' if dlt >= 0.02 else 'WEAK/redundant'}")

    # Gate-C preview: cross-site train(RUMC+PCNN) -> test(ZGT), if enough ZGT cases
    cen = df["center"].to_numpy()
    if (cen == "ZGT").sum() >= 30 and (np.isin(cen, ["RUMC", "PCNN"])).sum() >= 50:
        tr = np.isin(cen, ["RUMC", "PCNN"]); te = cen == "ZGT"
        print(f"\n=== Gate-C preview: train RUMC+PCNN (n={tr.sum()}) → test ZGT (n={te.sum()}, "
              f"pos={int(y[te].sum())}) ===")
        for name, X in [("clinical", Xc), ("radiomics", Xr), ("fusion_early", Xf)]:
            m = clone(pipe("logreg")); m.fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
            print(f"  {name:14s} ZGT AUROC {roc_auc_score(y[te], p):.3f} | ECE {ece(y[te], p):.3f}")
    else:
        print(f"\n(Gate-C cross-site deferred: need all folds; current ZGT n={(cen=='ZGT').sum()})")


if __name__ == "__main__":
    main()

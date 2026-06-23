#!/usr/bin/env python3
"""
scripts/picai_deep_gateb_shift.py — deep-arm Gate-B + shift-detection sharpness (steps 3-4).

Uses the DEEP OOF embeddings (data/processed/picai_deep_embeddings.npz) as the imaging arm and
compares against the radiomics arm (data/processed/picai_radiomics_features.csv):

  Gate-B (5x5 patient-level CV AUROC + ECE):
     deep imaging-only        vs radiomics imaging-only (0.773)
     deep + clinical fusion   vs radiomics + clinical fusion (0.804)
     (early-concat + late-mean; report the better fusion, matching picai_fusion.py conventions)

  Shift-detection sharpness:
     source(RUMC+PCNN) vs ZGT domain discriminator AUC on DEEP embeddings vs on radiomics.
     Sharper separation => MoRA's per-modality reliability signal is crisper on the deep arm.

The deep embeddings are out-of-fold; the downstream head/scaler are still fit per fold (Gate-B)
or via 5-fold CV on the discriminator (shift), so this is leakage-safe.

Run: ./.venv/bin/python scripts/picai_deep_gateb_shift.py
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

from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
OUT = os.path.join(ROOT, "results", "picai_deep")
REP, FOLDS = 5, 5


def ece(y, p, nb=15):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def pipe(kind):
    clf = (LogisticRegression(max_iter=2000, C=0.5) if kind == "logreg"
           else HistGradientBoostingClassifier(random_state=0))
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", clf)])


def cv_oof(X, y, kind, seed):
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = clone(pipe(kind)); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def cv(X, y, kind):
    a = [roc_auc_score(y, cv_oof(X, y, kind, s)) for s in range(REP)]
    return float(np.mean(a)), float(np.std(a)), ece(y, cv_oof(X, y, kind, 0))


def best_arm(X, y):
    return max((("logreg", cv(X, y, "logreg")), ("gboost", cv(X, y, "gboost"))), key=lambda t: t[1][0])


def disc_auc(Xsrc, Xzgt):
    """5-fold CV AUC of a source-vs-ZGT domain discriminator (higher => sharper detectable shift)."""
    X = np.vstack([Xsrc, Xzgt]); dom = np.r_[np.zeros(len(Xsrc)), np.ones(len(Xzgt))]
    d = cross_val_predict(Pipeline([("i", SimpleImputer(strategy="median")),
                                    ("s", StandardScaler()),
                                    ("c", LogisticRegression(max_iter=2000, C=0.5))]),
                          X, dom, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(dom, d))


def main():
    # deep embeddings frame
    z = np.load(os.path.join(PROC, "picai_deep_embeddings.npz"), allow_pickle=True)
    pids = [int(p) for p in z["__pid__"]]
    E = np.stack([z[str(p)] for p in pids]).astype(np.float64)
    emb_cols = [f"e{i}" for i in range(E.shape[1])]
    deep = pd.DataFrame(E, columns=emb_cols); deep.insert(0, "patient_id", pids)

    rad = pd.read_csv(os.path.join(PROC, "picai_radiomics_features.csv"))
    rad_cols = [c for c in rad.columns if c not in ("patient_id", "study_id")]

    mk = load_marksheet()
    mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    meta = mk[["patient_id", "label", "center"] + CLIN].drop_duplicates("patient_id")

    # align deep + radiomics + clinical on the SAME patient set (intersection)
    df = (deep.merge(rad[["patient_id"] + rad_cols], on="patient_id", how="inner")
              .merge(meta, on="patient_id", how="inner")).reset_index(drop=True)
    y = df["label"].to_numpy(int)
    cen = df["center"].to_numpy()
    Xc = df[CLIN].to_numpy(float)
    Xdeep = df[emb_cols].to_numpy(float)
    Xrad = df[rad_cols].to_numpy(float)
    print(f"[deep Gate-B] aligned n={len(df)} csPCa+={y.sum()} ({y.mean():.3f}) | "
          f"deep={len(emb_cols)}d radiomics={len(rad_cols)}d")
    print("per-center:", {c: int((cen == c).sum()) for c in ["RUMC", "PCNN", "ZGT"]})

    res = {"n": int(len(df)), "pos": int(y.sum())}

    # ---- Gate-B ----
    print("\n=== Gate-B: 5x5 CV AUROC (±sd) | ECE ===")
    rows = {}
    for name, X in [("clinical", Xc), ("radiomics", Xrad), ("deep", Xdeep)]:
        b = best_arm(X, y); rows[name] = b
        print(f"  {name:18s} {b[0]:7s} AUROC {b[1][0]:.3f} ± {b[1][1]:.3f} | ECE {b[1][2]:.3f}")

    # fusions (early-concat best-model + late-mean), report better per imaging arm
    def fusion(Xc, Xi, ck, ik):
        Xe = np.hstack([Xc, Xi])
        be = best_arm(Xe, y)
        la = [roc_auc_score(y, 0.5 * (cv_oof(Xc, y, ck, s) + cv_oof(Xi, y, ik, s))) for s in range(REP)]
        return be, (float(np.mean(la)), float(np.std(la)))

    ck = rows["clinical"][0]
    for arm, X, key in [("radiomics", Xrad, "radiomics"), ("deep", Xdeep, "deep")]:
        ik = rows[key][0]
        be, (lam, las) = fusion(Xc, X, ck, ik)
        best_fus = max(be[1][0], lam)
        rows[f"fusion_{arm}"] = best_fus
        print(f"  fusion({arm:9s})    early {be[1][0]:.3f} | late {lam:.3f} -> best {best_fus:.3f}")

    res["gateb"] = {
        "clinical_auroc": round(rows["clinical"][1][0], 4),
        "radiomics_imaging_auroc": round(rows["radiomics"][1][0], 4),
        "deep_imaging_auroc": round(rows["deep"][1][0], 4),
        "radiomics_fusion_auroc": round(rows["fusion_radiomics"], 4),
        "deep_fusion_auroc": round(rows["fusion_deep"], 4),
        "deep_imaging_ece": round(rows["deep"][1][2], 4),
        "radiomics_imaging_ece": round(rows["radiomics"][1][2], 4),
    }
    print(f"\n  deep imaging-only {rows['deep'][1][0]:.3f} vs radiomics 0.773 "
          f"(Δ {rows['deep'][1][0]-rows['radiomics'][1][0]:+.3f})")
    print(f"  deep fusion       {rows['fusion_deep']:.3f} vs radiomics-fusion "
          f"{rows['fusion_radiomics']:.3f} (Δ {rows['fusion_deep']-rows['fusion_radiomics']:+.3f})")

    # ---- shift-detection sharpness ----
    src = np.isin(cen, ["RUMC", "PCNN"]); zgt = cen == "ZGT"
    a_deep = disc_auc(Xdeep[src], Xdeep[zgt])
    a_rad = disc_auc(Xrad[src], Xrad[zgt])
    a_clin = disc_auc(Xc[src], Xc[zgt])
    print(f"\n=== shift-detection: source(RUMC+PCNN) vs ZGT discriminator AUC ===")
    print(f"  deep embeddings  {a_deep:.3f}")
    print(f"  radiomics        {a_rad:.3f}  (reference ~0.83)")
    print(f"  clinical         {a_clin:.3f}")
    res["shift_discriminator_auc"] = {"deep": round(a_deep, 4), "radiomics": round(a_rad, 4),
                                      "clinical": round(a_clin, 4)}

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "gateb_shift.json"), "w"), indent=2)
    print(f"\nwrote {OUT}/gateb_shift.json")


if __name__ == "__main__":
    main()

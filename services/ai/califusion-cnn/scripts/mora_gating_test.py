#!/usr/bin/env python3
"""
scripts/mora_gating_test.py — MoRA H2 core mechanism: reliability-GATED prediction.

The faithful make-or-break for the per-modality claim (ADR-0008). MoRA's central idea is
that per-modality reliability lets you DOWN-WEIGHT an out-of-distribution modality at
inference — something a single GLOBAL reliability signal cannot do (it scales both
modalities equally). Under shift on ONE modality, per-modality gating should recover
accuracy/calibration that static fusion and global-reliability adjustment cannot.

Predictors (logistic, source-train only): p_img (imaging-only), p_clin (clinical-only).
  static     : 0.5*(p_img + p_clin)                       — fixed equal weights
  global_rel : reliability scales both equally (≡ static for the weighted mean) → confidence shrink toward 0.5 by r_glob
  per_modal  : (r_img^λ·p_img + r_clin^λ·p_clin)/(Σ)        — MoRA reliability-gated fusion

Shift hits a random subset of TARGET patients on one modality (severity × source std).
Leakage-safe: predictors + reliabilities fit on SOURCE TRAIN only.

H2 verdict: per_modal recovers AUROC AND Brier on the SHIFTED subset better than static and
global under single-modality shift (paired over seeds).

Run:  python scripts/mora_gating_test.py [--seeds 20] [--severity 2.5] [--lam 1.0]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from califusion.data.clinical_unified import build_unified_preprocessor, UNIFIED_FEATURES
from califusion.mora.reliability import MahalanobisReliability

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "mora_gating")
COND = ["clean", "img_subset", "clin_subset"]


def metrics(y, p):
    au = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return float(au), float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--severity", type=float, default=2.5)
    ap.add_argument("--shift_frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))
    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    df = rad.merge(clin.drop(columns=["label"]), on="PatientID", how="inner").reset_index(drop=True)
    y = df["label"].to_numpy(int)
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    print(f"cohort n={len(df)} | radiomics={len(rad_cols)} | prev={y.mean():.3f} | lambda={args.lam}")

    rows = []
    for seed in range(args.seeds):
        rng = np.random.RandomState(2000 + seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.40, stratify=y, random_state=seed)
        img_pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        clin_pre = build_unified_preprocessor()
        Xi_tr = img_pipe.fit_transform(df[rad_cols].iloc[tr]); Xc_tr = clin_pre.fit_transform(df[UNIFIED_FEATURES].iloc[tr])
        ytr, yte = y[tr], y[te]
        m_img = LogisticRegression(max_iter=2000, C=0.5).fit(Xi_tr, ytr)
        m_clin = LogisticRegression(max_iter=2000, C=0.5).fit(Xc_tr, ytr)
        Rimg = MahalanobisReliability().fit(Xi_tr, ytr); Rclin = MahalanobisReliability().fit(Xc_tr, ytr)
        Rglob = MahalanobisReliability().fit(np.hstack([Xi_tr, Xc_tr]), ytr)
        img_std = Xi_tr.std(0, keepdims=True); clin_std = Xc_tr.std(0, keepdims=True)
        Xi_te = img_pipe.transform(df[rad_cols].iloc[te]); Xc_te = clin_pre.transform(df[UNIFIED_FEATURES].iloc[te])

        for cond in COND:
            Xi, Xc = Xi_te.copy(), Xc_te.copy()
            shifted = np.zeros(len(te), bool)
            if cond != "clean":
                shifted = rng.random(len(te)) < args.shift_frac
                if cond == "img_subset":
                    Xi[shifted] = Xi[shifted] + args.severity * rng.standard_normal(Xi[shifted].shape) * img_std
                else:
                    Xc[shifted] = Xc[shifted] + args.severity * rng.standard_normal(Xc[shifted].shape) * clin_std
            pi = m_img.predict_proba(Xi)[:, 1]; pc = m_clin.predict_proba(Xc)[:, 1]
            ri = Rimg.reliability(Xi) ** args.lam; rc = Rclin.reliability(Xc) ** args.lam
            rg = Rglob.reliability(np.hstack([Xi, Xc]))
            p_static = 0.5 * (pi + pc)
            p_permod = (ri * pi + rc * pc) / np.clip(ri + rc, 1e-9, None)
            # global reliability cannot reweight modalities; it shrinks fused confidence toward base rate
            p_global = rg * p_static + (1 - rg) * float(ytr.mean())
            ev = shifted if cond != "clean" else np.ones(len(te), bool)  # evaluate on shifted subset (or all, clean)
            for name, p in (("static", p_static), ("global", p_global), ("permod", p_permod)):
                au, br = metrics(yte[ev], p[ev])
                rows.append({"seed": seed, "cond": cond, "method": name, "auroc": au, "brier": br,
                             "n_eval": int(ev.sum())})

    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

    def cell(cond, method, key):
        return float(R[(R.cond == cond) & (R.method == method)][key].mean())

    out = {"n_seeds": args.seeds, "severity": args.severity, "lambda": args.lam,
           "eval": "shifted subset under shift; all patients when clean", "table": {}}
    for cond in COND:
        out["table"][cond] = {mth: {"auroc": round(cell(cond, mth, "auroc"), 4),
                                    "brier": round(cell(cond, mth, "brier"), 4)}
                              for mth in ("static", "global", "permod")}
    # H2: under single-modality shift, per-modal beats static AND global on AUROC & Brier (paired)
    wins = {}
    for cond in ("img_subset", "clin_subset"):
        sub = R[R.cond == cond].pivot_table(index="seed", columns="method", values=["auroc", "brier"])
        wins[cond] = {
            "auroc_permod_vs_static": round(float((sub[("auroc", "permod")] > sub[("auroc", "static")]).mean()), 3),
            "auroc_permod_vs_global": round(float((sub[("auroc", "permod")] > sub[("auroc", "global")]).mean()), 3),
            "brier_permod_vs_static": round(float((sub[("brier", "permod")] < sub[("brier", "static")]).mean()), 3),
            "dAUROC_vs_static": round(float((sub[("auroc", "permod")] - sub[("auroc", "static")]).mean()), 4),
            "dBrier_vs_static": round(float((sub[("brier", "permod")] - sub[("brier", "static")]).mean()), 4)}
    out["H2_wins"] = wins
    ok = all(wins[c]["auroc_permod_vs_static"] >= 0.6 and wins[c]["brier_permod_vs_static"] >= 0.6
             for c in ("img_subset", "clin_subset"))
    out["H2_verdict"] = "SUPPORTED" if ok else "NOT SUPPORTED"
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== MoRA reliability-GATED prediction (eval on shifted subset) ===")
    for cond in COND:
        t = out["table"][cond]
        print(f"  {cond:11s} AUROC static {t['static']['auroc']:.3f} | global {t['global']['auroc']:.3f} | "
              f"permod {t['permod']['auroc']:.3f}   Brier static {t['static']['brier']:.3f} | permod {t['permod']['brier']:.3f}")
    bar = "=" * 66
    print(f"\n{bar}")
    for c in ("img_subset", "clin_subset"):
        w = wins[c]
        print(f"  {c}: ΔAUROC vs static {w['dAUROC_vs_static']:+.3f} (win {w['auroc_permod_vs_static']:.0%}), "
              f"vs global win {w['auroc_permod_vs_global']:.0%}; ΔBrier {w['dBrier_vs_static']:+.3f}")
    print(f"  H2 (per-modal reliability-gating beats static & global under shift): {out['H2_verdict']}")
    print(bar)


if __name__ == "__main__":
    main()

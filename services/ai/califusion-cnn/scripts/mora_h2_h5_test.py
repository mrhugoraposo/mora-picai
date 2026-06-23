#!/usr/bin/env python3
"""
scripts/mora_h2_h5_test.py — MoRA make-or-break test (ADR-0008, RESEARCH_DESIGN H2/H5).

Decisive, cheap experiment on the matched tabular cohort (radiomics imaging ⊕ clinical).
Selective prediction defers by CONFIDENCE modulated by reliability (the actual MoRA claim:
reliability down-weights confidence on OOD cases). Shift hits a *random subset* of target
patients on one modality (realistic: a new scanner / missing clinical for some patients),
so reliability has within-test signal to localize the affected cases.

  H2 — under modality-specific shift, does CONF × PER-MODALITY reliability beat
       CONF × GLOBAL reliability (and beat CONF-only) for risk-controlled deferral
       (lower AURC / selective risk)?
  H5 — among deferred-and-truly-shifted cases, does the lower per-modality reliability
       point at the degraded modality?

Leakage-safe: transformers, base predictor, operating threshold, and Mahalanobis
reliabilities are all fit on SOURCE TRAIN only; shift applied to TARGET subset only.

Honest caveat reported in output: base AUROC ≈0.58 caps how much ANY selective predictor
can gain — we report the per-modality-vs-global delta, not an absolute selective-risk claim.

Run:  python scripts/mora_h2_h5_test.py [--seeds 20] [--severity 2.5] [--shift_frac 0.5]
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
from sklearn.metrics import roc_auc_score

from califusion.data.clinical_unified import build_unified_preprocessor, UNIFIED_FEATURES
from califusion.mora.reliability import MahalanobisReliability, combine_reliability
from califusion.eval.metrics import youden_threshold

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "mora_h2_h5")
COND = ["clean", "img_subset", "clin_subset"]
SHIFT = ["img_subset", "clin_subset"]


def aurc(score, err):
    """Area under risk–coverage (retain highest-score first); lower = better."""
    e = err[np.argsort(-score)].astype(float)
    return float((np.cumsum(e) / (np.arange(len(e)) + 1)).mean())


def selective_risk(score, err, coverage):
    n = len(score); k = max(1, int(round(coverage * n)))
    return float(err[np.argsort(-score)[:k]].mean())


def add_noise(X, std, severity, rng):
    return X + severity * rng.standard_normal(X.shape) * std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--severity", type=float, default=2.5)
    ap.add_argument("--shift_frac", type=float, default=0.5)
    ap.add_argument("--coverages", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()

    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))
    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    df = rad.merge(clin.drop(columns=["label"]), on="PatientID", how="inner").reset_index(drop=True)
    y = df["label"].to_numpy(int)
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    print(f"cohort n={len(df)} | radiomics={len(rad_cols)} clinical={UNIFIED_FEATURES} | prev={y.mean():.3f}")

    rows = []
    for seed in range(args.seeds):
        rng = np.random.RandomState(1000 + seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.40, stratify=y, random_state=seed)
        img_pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        clin_pre = build_unified_preprocessor()
        Xi_tr = img_pipe.fit_transform(df[rad_cols].iloc[tr]); Xc_tr = clin_pre.fit_transform(df[UNIFIED_FEATURES].iloc[tr])
        ytr, yte = y[tr], y[te]
        Z_tr = np.hstack([Xi_tr, Xc_tr])
        base = LogisticRegression(max_iter=2000, C=0.5).fit(Z_tr, ytr)
        thr = youden_threshold(ytr, base.predict_proba(Z_tr)[:, 1])
        Rimg = MahalanobisReliability().fit(Xi_tr, ytr)
        Rclin = MahalanobisReliability().fit(Xc_tr, ytr)
        Rglob = MahalanobisReliability().fit(Z_tr, ytr)
        img_std = Xi_tr.std(0, keepdims=True); clin_std = Xc_tr.std(0, keepdims=True)
        Xi_te = img_pipe.transform(df[rad_cols].iloc[te]); Xc_te = clin_pre.transform(df[UNIFIED_FEATURES].iloc[te])

        for cond in COND:
            Xi, Xc = Xi_te.copy(), Xc_te.copy()
            shifted = np.zeros(len(te), bool)
            if cond != "clean":
                shifted = rng.random(len(te)) < args.shift_frac
                if cond == "img_subset":
                    Xi[shifted] = add_noise(Xi[shifted], img_std, args.severity, rng)
                else:
                    Xc[shifted] = add_noise(Xc[shifted], clin_std, args.severity, rng)
            Z = np.hstack([Xi, Xc])
            p = base.predict_proba(Z)[:, 1]
            err = ((p >= thr).astype(int) != yte).astype(float)
            conf = np.abs(p - 0.5)                       # selective-prediction confidence
            r_img = Rimg.reliability(Xi); r_clin = Rclin.reliability(Xc); r_glob = Rglob.reliability(Z)
            r_permod = combine_reliability([r_img, r_clin], how="min")
            s_conf = conf                                 # confidence-only deferral (baseline)
            s_permod = conf * r_permod                    # MoRA: per-modality reliability-modulated
            s_global = conf * r_glob                      # ablation: single global reliability
            rec = {"seed": seed, "cond": cond,
                   "auroc": float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else np.nan,
                   "err_rate": float(err.mean()),
                   "aurc_conf": aurc(s_conf, err), "aurc_permod": aurc(s_permod, err),
                   "aurc_global": aurc(s_global, err), "aurc_random": aurc(rng.random(len(err)), err)}
            for c in args.coverages:
                rec[f"sr_conf@{c}"] = selective_risk(s_conf, err, c)
                rec[f"sr_permod@{c}"] = selective_risk(s_permod, err, c)
                rec[f"sr_global@{c}"] = selective_risk(s_global, err, c)
            if cond != "clean":
                k = max(1, int(0.30 * len(te)))
                deferred = np.argsort(s_permod)[:k]       # most-deferred 30%
                ds = deferred[shifted[deferred]]          # deferred AND truly shifted
                rec["deferred_shift_recall"] = float(shifted[deferred].mean())
                rec["attribution_acc"] = (float(((r_img[ds] <= r_clin[ds]) == (cond == "img_subset")).mean())
                                          if len(ds) else np.nan)
            rows.append(rec)

    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

    def m(cond, key):
        return float(R[R.cond == cond][key].mean())

    sub = R[R.cond.isin(SHIFT)]
    h2_pm, h2_gl, h2_cf = sub.aurc_permod.mean(), sub.aurc_global.mean(), sub.aurc_conf.mean()
    h2_win_vs_global = float((sub.aurc_permod < sub.aurc_global).mean())
    h2_win_vs_conf = float((sub.aurc_permod < sub.aurc_conf).mean())
    h5 = float(sub["attribution_acc"].mean())
    h5_recall = float(sub["deferred_shift_recall"].mean())

    summary = {
        "n_seeds": args.seeds, "severity": args.severity, "shift_frac": args.shift_frac,
        "base_auroc_clean": round(m("clean", "auroc"), 4),
        "honest_caveat": "base AUROC ~0.58 caps absolute selective-prediction gains; H2 tests the "
                         "per-modality-vs-global DELTA, not an absolute risk claim.",
        "err_rate": {c: round(m(c, "err_rate"), 4) for c in COND},
        "AURC": {c: {"conf_only": round(m(c, "aurc_conf"), 4),
                     "permod": round(m(c, "aurc_permod"), 4),
                     "global": round(m(c, "aurc_global"), 4),
                     "random": round(m(c, "aurc_random"), 4)} for c in COND},
        "H2": {"aurc_permod": round(float(h2_pm), 4), "aurc_global": round(float(h2_gl), 4),
               "aurc_conf_only": round(float(h2_cf), 4),
               "permod_beats_global_winrate": round(h2_win_vs_global, 3),
               "permod_beats_confonly_winrate": round(h2_win_vs_conf, 3),
               "verdict": "SUPPORTED" if (h2_pm < h2_gl and h2_win_vs_global >= 0.6
                                          and h2_pm < h2_cf) else "WEAK/NOT SUPPORTED"},
        "H5": {"attribution_acc": round(h5, 3), "deferred_shift_recall": round(h5_recall, 3),
               "verdict": "SUPPORTED" if h5 >= 0.7 else "NOT SUPPORTED"},
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== MoRA H2/H5 (confidence×reliability deferral, subset shift) ===")
    print(f"base AUROC(clean)={summary['base_auroc_clean']} | err {summary['err_rate']}")
    print("AURC (lower=better):")
    for c in COND:
        a = summary["AURC"][c]
        print(f"  {c:11s} conf {a['conf_only']:.3f} | permod {a['permod']:.3f} | "
              f"global {a['global']:.3f} | random {a['random']:.3f}")
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"H2: per-mod AURC {summary['H2']['aurc_permod']:.3f} vs global {summary['H2']['aurc_global']:.3f} "
          f"vs conf-only {summary['H2']['aurc_conf_only']:.3f}")
    print(f"    win vs global {summary['H2']['permod_beats_global_winrate']:.0%}, "
          f"vs conf-only {summary['H2']['permod_beats_confonly_winrate']:.0%} -> {summary['H2']['verdict']}")
    print(f"H5: attribution acc {summary['H5']['attribution_acc']:.0%} "
          f"(deferred-shift recall {summary['H5']['deferred_shift_recall']:.0%}) -> {summary['H5']['verdict']}")
    print(bar)


if __name__ == "__main__":
    main()

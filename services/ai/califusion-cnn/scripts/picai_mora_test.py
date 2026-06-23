#!/usr/bin/env python3
"""
scripts/picai_mora_test.py — THE make-or-break: MoRA on PI-CAI's REAL vendor shift (ADR-0009).

Unlike the Lung1 tests (synthetic shift, redundant modalities), this uses the AUTHENTIC
cross-site shift: train on RUMC+PCNN (Siemens) → test on ZGT (Philips). The imaging modality
(MRI radiomics) genuinely goes out-of-distribution there (fold0 preview: radiomics 0.73→0.57,
while clinical holds), and naive fusion is dragged below clinical. The question (H2):

  Does PER-MODALITY reliability detect that the imaging modality is unreliable on ZGT,
  down-weight it, and recover accuracy/calibration — better than a single GLOBAL reliability
  signal (which cannot selectively down-weight) and better than static fusion?

Per-modality predictors + Mahalanobis reliabilities fit on SOURCE only; applied to the ZGT
target (no target labels used to fit anything). Reports AUROC/Brier/ECE on ZGT for static /
global / per-modal, the reliability values (r_img should drop on ZGT), and bootstrap CIs.

Runs on whatever is extracted (fold0 = underpowered ZGT n~76 preview; full 5 folds = ZGT 80 pos,
the POWERED make-or-break). Run: python scripts/picai_mora_test.py
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from califusion.data.picai_clinical import load_marksheet, NUMERIC as CLIN
from califusion.mora.reliability import MahalanobisReliability

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "picai_mora")
EPS = 1e-6


def ece(y, p, nb=12):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def fit_prob(X, y):
    pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, C=0.5))]).fit(X, y)
    return pipe


def prep(Xtr, Xte):
    p = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)


def boot_ci(y, pa, pb, n=2000):
    rng = np.random.RandomState(0); d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    rad = pd.read_csv(os.path.join(PROC, "picai_radiomics_features.csv"))
    mk = load_marksheet(); mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    df = rad.merge(mk[["patient_id", "label", "center"] + CLIN], on="patient_id", how="inner").reset_index(drop=True)
    rad_cols = [c for c in rad.columns if c not in ("patient_id", "study_id")]
    src = df["center"].isin(["RUMC", "PCNN"]).to_numpy(); tgt = (df["center"] == "ZGT").to_numpy()
    ys, yt = df["label"].to_numpy()[src], df["label"].to_numpy()[tgt]
    print(f"source(RUMC+PCNN) n={src.sum()} pos={ys.sum()} | target(ZGT, real Philips shift) n={tgt.sum()} pos={yt.sum()}")
    if tgt.sum() < 20:
        print("target too small; need more folds"); return

    Xi, Xc = df[rad_cols], df[CLIN]
    # per-modality predictors (source-fit)
    m_img = fit_prob(Xi[src], ys); m_clin = fit_prob(Xc[src], ys)
    pi_t = m_img.predict_proba(Xi[tgt])[:, 1]; pc_t = m_clin.predict_proba(Xc[tgt])[:, 1]
    # reliabilities (source-fit), transformed feature space
    Xi_s, Xi_t = prep(Xi[src], Xi[tgt]); Xc_s, Xc_t = prep(Xc[src], Xc[tgt])
    Ri = MahalanobisReliability().fit(Xi_s, ys); Rc = MahalanobisReliability().fit(Xc_s, ys)
    Rg = MahalanobisReliability().fit(np.hstack([Xi_s, Xc_s]), ys)
    ri = Ri.reliability(Xi_t); rc = Rc.reliability(Xc_t); rg = Rg.reliability(np.hstack([Xi_t, Xc_t]))
    # reliability sanity: source self-reliability vs target
    ri_src = Ri.reliability(Xi_s).mean(); rc_src = Rc.reliability(Xc_s).mean()

    base = float(ys.mean())
    p_static = 0.5 * (pi_t + pc_t)
    p_permod = (ri * pi_t + rc * pc_t) / np.clip(ri + rc, EPS, None)
    p_global = rg * p_static + (1 - rg) * base

    def row(name, p):
        au = roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else np.nan
        return name, float(au), brier_score_loss(yt, np.clip(p, EPS, 1 - EPS)), ece(yt, p)

    res = [row("clinical-only", pc_t), row("imaging-only", pi_t),
           row("static fusion", p_static), row("global-reliability", p_global),
           row("per-modal (MoRA)", p_permod)]
    print("\n=== ZGT (real vendor shift) — AUROC | Brier | ECE ===")
    for n, a, b, e in res:
        print(f"  {n:20s} AUROC {a:.3f} | Brier {b:.3f} | ECE {e:.3f}")
    print(f"\nReliability detection: r_img source {ri_src:.2f} → ZGT {ri.mean():.2f}  "
          f"(drop = imaging detected OOD); r_clin source {rc_src:.2f} → ZGT {rc.mean():.2f}")
    lo, hi = boot_ci(yt, p_permod, p_static)
    print(f"per-modal − static ΔAUROC on ZGT = {res[4][1]-res[2][1]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    h2 = (res[4][1] > res[2][1]) and (res[4][1] >= res[3][1])
    print(f"\nH2 (per-modal recovers under real shift > static & global): "
          f"{'SUPPORTED (preview)' if h2 else 'NOT in this slice'}  "
          f"[{'UNDERPOWERED — fold0 only' if tgt.sum()<60 else 'powered'}]")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"n_target": int(tgt.sum()), "pos_target": int(yt.sum()),
               "results": {n: {"auroc": round(a, 4), "brier": round(b, 4), "ece": round(e, 4)} for n, a, b, e in res},
               "r_img_source": round(float(ri_src), 3), "r_img_zgt": round(float(ri.mean()), 3),
               "r_clin_source": round(float(rc_src), 3), "r_clin_zgt": round(float(rc.mean()), 3),
               "permod_minus_static_auroc": round(res[4][1] - res[2][1], 4), "ci": [round(lo, 4), round(hi, 4)],
               "h2_preview": bool(h2), "powered": bool(tgt.sum() >= 60)},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

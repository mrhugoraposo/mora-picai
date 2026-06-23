#!/usr/bin/env python3
"""
scripts/picai_sota_baselines_deep.py — the make-or-break: MoRA vs named SOTA under modality
failure, using the DEEP bpMRI embeddings as the imaging arm (vs the radiomics version in
scripts/picai_sota_baselines.py).

Identical protocol to picai_sota_baselines.py (so results are directly comparable):
  imaging broken on a random 50% of TEST patients (additive Gaussian, `severity`); compare
    clinical / static / TransCal-CPCS / evidential-TMC / MoRA (per-modality discriminator
    reliability gating, calibrated r=clip((1-d)/(1-d_ref),0,1)^λ) + weighted-conformal vs
    MoRA-deferral selective risk.
The ONLY change: imaging features = deep OOF embeddings (data/processed/picai_deep_embeddings.npz)
instead of radiomics. The embeddings are out-of-fold (each case embedded by a model that never
trained on it) so swapping them in is leakage-safe; the per-fold imaging head still fits on the
train split only.

Run: ./.venv/bin/python scripts/picai_sota_baselines_deep.py [--seeds 10] [--severity 4]
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
import torch  # noqa: F401  (used by EDL import path parity)
import torch.nn as nn  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar  # noqa: F401

from califusion.data.picai_clinical import load_marksheet, NUMERIC as CLIN

# reuse the exact SOTA implementations (EDL/TMC, iw-temperature, disc reliability, etc.)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "picai_sota_baselines", os.path.join(os.path.dirname(__file__), "picai_sota_baselines.py"))
_S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_S)
ece = _S.ece
iw_temperature = _S.iw_temperature
train_edl = _S.train_edl; edl_alpha = _S.edl_alpha; ds_combine = _S.ds_combine
disc_reliability = _S.disc_reliability
selective_risk = _S.selective_risk
_logit = _S._logit; _sig = _S._sig

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "picai_deep")
EMB = os.path.join(PROC, "picai_deep_embeddings.npz")
EPS = 1e-6


def load_deep_frame():
    """Return (df with patient_id+label+CLIN+embedding cols, emb_cols, y)."""
    z = np.load(EMB, allow_pickle=True)
    pids = [int(p) for p in z["__pid__"]]
    E = np.stack([z[str(p)] for p in pids]).astype(np.float64)   # [N,512] OOF embeddings
    emb_cols = [f"e{i}" for i in range(E.shape[1])]
    ed = pd.DataFrame(E, columns=emb_cols); ed.insert(0, "patient_id", pids)
    mk = load_marksheet()
    mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    df = ed.merge(mk[["patient_id", "label"] + CLIN], on="patient_id", how="inner").reset_index(drop=True)
    return df, emb_cols, df["label"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--severity", type=float, default=4.0)
    ap.add_argument("--lam", type=float, default=2.0)
    args = ap.parse_args()

    df, emb_cols, y = load_deep_frame()
    print(f"[deep-SOTA] PI-CAI n={len(df)} csPCa+={y.sum()} | imaging=DEEP {len(emb_cols)}-d "
          f"| broken on 50% of test, severity {args.severity}")

    methods = ["clinical", "static", "TransCal/CPCS", "evidential/TMC", "MoRA"]
    auc = {m: [] for m in methods}; ec = {m: [] for m in methods}
    sr_conf, sr_mora = [], []
    for seed in range(args.seeds):
        rng = np.random.RandomState(seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
        si = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]).fit(df[emb_cols].iloc[tr])
        sc = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]).fit(df[CLIN].iloc[tr])
        Xi_tr, Xi_te = si.transform(df[emb_cols].iloc[tr]), si.transform(df[emb_cols].iloc[te])
        Xc_tr, Xc_te = sc.transform(df[CLIN].iloc[tr]), sc.transform(df[CLIN].iloc[te])
        ytr, yte = y[tr], y[te]
        mi = LogisticRegression(max_iter=2000, C=0.5).fit(Xi_tr, ytr)
        mc = LogisticRegression(max_iter=2000, C=0.5).fit(Xc_tr, ytr)
        # break imaging on 50% of test (same additive-Gaussian corruption as the radiomics harness)
        broken = rng.rand(len(te)) < 0.5
        Xi_te_c = Xi_te.copy(); Xi_te_c[broken] += args.severity * rng.standard_normal(Xi_te_c[broken].shape)
        pi = mi.predict_proba(Xi_te_c)[:, 1]; pc = mc.predict_proba(Xc_te)[:, 1]
        static = 0.5 * (pi + pc)
        err = ((static >= 0.5).astype(int) != yte).astype(float)

        auc["clinical"].append(roc_auc_score(yte, pc)); ec["clinical"].append(ece(yte, pc))
        auc["static"].append(roc_auc_score(yte, static)); ec["static"].append(ece(yte, static))

        # TransCal/CPCS — iw temperature on fused logits (joint discriminator weights)
        zf = _logit(static)
        Z = np.vstack([np.hstack([Xi_tr, Xc_tr]), np.hstack([Xi_te_c, Xc_te])])
        dom = np.r_[np.zeros(len(tr)), np.ones(len(te))]
        dj = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Z, dom, cv=5, method="predict_proba")[:, 1]
        w_tr = np.clip(dj[:len(tr)] / (1 - dj[:len(tr)] + EPS), EPS, 50)
        zf_tr = _logit(0.5 * (mi.predict_proba(Xi_tr)[:, 1] + mc.predict_proba(Xc_tr)[:, 1]))
        T = iw_temperature(zf_tr, ytr, w_tr)
        p_tc = _sig(zf / T)
        auc["TransCal/CPCS"].append(roc_auc_score(yte, p_tc)); ec["TransCal/CPCS"].append(ece(yte, p_tc))

        # Evidential / TMC — Dirichlet EDL per modality + DS combine
        ni = train_edl(Xi_tr, ytr, seed=seed); nc = train_edl(Xc_tr, ytr, seed=seed)
        ai = edl_alpha(ni, Xi_te_c); ac = edl_alpha(nc, Xc_te)
        p_ev = ds_combine(ai, ac)[:, 1]
        auc["evidential/TMC"].append(roc_auc_score(yte, p_ev)); ec["evidential/TMC"].append(ece(yte, p_ev))

        # MoRA — per-modality discriminator reliability gating
        ri, wi_ratio = disc_reliability(Xi_tr, Xi_te_c, args.lam); rc, _ = disc_reliability(Xc_tr, Xc_te, args.lam)
        pm = (ri * pi + rc * pc) / (ri + rc)
        auc["MoRA"].append(roc_auc_score(yte, pm)); ec["MoRA"].append(ece(yte, pm))

        # selective prediction: weighted-conformal (global joint weight) vs MoRA per-modality reliability
        w_global = np.clip(dj[len(tr):] / (1 - dj[len(tr):] + EPS), EPS, 50)
        conf = np.abs(static - 0.5)
        sr_conf.append(selective_risk(conf / (w_global + EPS), err, 0.8))
        sr_mora.append(selective_risk(conf * np.minimum(ri, rc), err, 0.8))

    def stat(d, k): return float(np.mean(d[k])), float(np.std(d[k]))
    print("\n=== AUROC under modality failure (DEEP imaging arm, mean±sd, %d seeds) ===" % args.seeds)
    for m in methods:
        a, s = stat(auc, m)
        print(f"  {m:16s} AUROC {a:.3f}±{s:.3f} | ECE {stat(ec, m)[0]:.3f}")
    mora_a = np.array(auc["MoRA"])
    print("\n  MoRA vs each (ΔAUROC, win-rate over seeds):")
    for m in ["static", "TransCal/CPCS", "evidential/TMC"]:
        o = np.array(auc[m]); print(f"    vs {m:16s} Δ {mora_a.mean()-o.mean():+.3f}  win {np.mean(mora_a>o):.0%}")
    print(f"\n  Selective risk @80% coverage (lower=better): weighted-conformal {np.mean(sr_conf):.3f} | "
          f"MoRA-deferral {np.mean(sr_mora):.3f}  (MoRA better in {np.mean(np.array(sr_mora)<np.array(sr_conf)):.0%} of seeds)")

    os.makedirs(OUT, exist_ok=True)
    res = {"arm": "deep", "severity": args.severity, "seeds": args.seeds, "lam": args.lam,
           "auroc": {m: round(stat(auc, m)[0], 4) for m in methods},
           "auroc_sd": {m: round(stat(auc, m)[1], 4) for m in methods},
           "ece": {m: round(stat(ec, m)[0], 4) for m in methods},
           "mora_delta_winrate": {m: [round(mora_a.mean() - np.array(auc[m]).mean(), 4),
                                      round(float(np.mean(mora_a > np.array(auc[m]))), 3)]
                                  for m in ["static", "TransCal/CPCS", "evidential/TMC"]},
           "selective_risk_80": {"weighted_conformal": round(float(np.mean(sr_conf)), 4),
                                 "mora_deferral": round(float(np.mean(sr_mora)), 4)}}
    out_path = os.path.join(OUT, f"sota_deep_sev{int(args.severity)}.json")
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

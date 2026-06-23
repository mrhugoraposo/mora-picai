#!/usr/bin/env python3
"""
scripts/picai_sota_baselines.py — does MoRA beat NAMED SOTA under modality failure? (the paper gate)

On PI-CAI (powered), with imaging broken on a random 50% of TEST patients, compare MoRA's per-modality
test-time reliability gating against faithful SOTA reliability baselines (no strawmen):
  - static fusion            — naive 0.5*(p_img+p_clin), trusts the broken scan
  - TransCal/CPCS            — importance-weighted temperature (global calibration; Park'20/Wang'20).
                               Monotonic in the fused score → AUROC ≡ static (the point: global calibration
                               recalibrates but CANNOT recover discrimination); we report its ECE gain.
  - Evidential / TMC (Han'21)— genuine Dirichlet EDL per modality + Dempster-Shafer combination. Down-weights
                               uncertain views, BUT evidence is learned on SOURCE → may not flag novel failure.
  - weighted-conformal (Tib'19) vs MoRA-attributed deferral — selective risk @ coverage (global weight vs per-modality).
  - MoRA                     — per-modality domain-discriminator reliability gating (test-time, label-free).

Thesis: only test-time PER-MODALITY shift detection recovers discrimination under failure; global calibration/
conformal can't reweight modalities, and source-trained evidential uncertainty doesn't adapt to novel failure.

Run: python scripts/picai_sota_baselines.py [--seeds 10] [--severity 4]
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
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar

from califusion.data.picai_clinical import load_marksheet, NUMERIC as CLIN

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "picai_sota")
EPS = 1e-6


def ece(y, p, nb=12):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def fit_lr(X, y):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, C=0.5))]).fit(X, y)


def _logit(p): return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
def _sig(z): return 1.0 / (1.0 + np.exp(-z))


def iw_temperature(z_val, y_val, w_val):
    """CPCS/TransCal: temperature minimising importance-weighted NLL (global calibration)."""
    w = w_val / (w_val.mean() + EPS)
    def f(T):
        p = np.clip(_sig(z_val / max(T, 1e-3)), EPS, 1 - EPS)
        return -np.mean(w * (y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
    return float(minimize_scalar(f, bounds=(0.05, 20), method="bounded").x)


# ---------------- Evidential / Trusted Multi-View (Han 2021) ----------------
class EDL(nn.Module):
    def __init__(self, d, h=32, K=2):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, K))

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))  # evidence >= 0


def edl_loss(evi, y1h, lam):
    alpha = evi + 1; S = alpha.sum(1, keepdim=True); p = alpha / S
    err = ((y1h - p) ** 2).sum(1)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1)
    a_t = y1h + (1 - y1h) * alpha                       # remove true-class evidence for KL
    St = a_t.sum(1, keepdim=True); K = y1h.shape[1]
    kl = (torch.lgamma(St).squeeze(1) - torch.lgamma(a_t).sum(1)
          - torch.lgamma(torch.tensor(float(K))) + torch.lgamma(torch.ones(1)).sum()
          + ((a_t - 1) * (torch.digamma(a_t) - torch.digamma(St))).sum(1))
    return (err + var + lam * kl).mean()


def train_edl(X, y, epochs=120, seed=0):
    torch.manual_seed(seed)
    net = EDL(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32); y1h = torch.tensor(np.c_[1 - y, y], dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad(); loss = edl_loss(net(Xt), y1h, min(1.0, ep / 40)); loss.backward(); opt.step()
    return net


def edl_alpha(net, X):
    with torch.no_grad():
        return (net(torch.tensor(X, dtype=torch.float32)) + 1).numpy()


def ds_combine(a1, a2, K=2):
    """Dempster-Shafer combination of two Dirichlets (TMC, Han 2021)."""
    S1 = a1.sum(1, keepdims=True); S2 = a2.sum(1, keepdims=True)
    b1 = (a1 - 1) / S1; u1 = K / S1; b2 = (a2 - 1) / S2; u2 = K / S2
    bb = b1[:, :, None] * b2[:, None, :]
    C = (bb.sum((1, 2)) - np.einsum('nkk->n', bb))[:, None]
    b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - C + EPS)
    u = (u1 * u2) / (1 - C + EPS)
    S = K / u; alpha = b * S + 1
    return alpha / alpha.sum(1, keepdims=True)


def disc_reliability(Xs, Xt, lam):
    """Per-modality reliability: r≈1 when as in-distribution as a typical source point, →0 as OOD.
    Calibrated against the discriminator's in-distribution level d_ref (median on source)."""
    X = np.vstack([Xs, Xt]); dom = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    d = cross_val_predict(Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=0.5))]),
                          X, dom, cv=5, method="predict_proba")[:, 1]
    d_ref = np.median(d[:len(Xs)])                      # in-distribution discriminator level
    d_t = d[len(Xs):]
    r = np.clip((1 - d_t) / (1 - d_ref + EPS), 0.0, 1.0) ** lam   # ≈1 in-dist, →0 OOD (calibrated)
    return r, np.clip(d_t / (1 - d_t + EPS), EPS, 50)


def selective_risk(score, err, cov):
    k = max(1, int(round(cov * len(score))))
    return float(err[np.argsort(-score)[:k]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--severity", type=float, default=4.0)
    ap.add_argument("--lam", type=float, default=2.0)
    args = ap.parse_args()

    rad = pd.read_csv(os.path.join(PROC, "picai_radiomics_features.csv"))
    mk = load_marksheet(); mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    df = rad.merge(mk[["patient_id", "label"] + CLIN], on="patient_id", how="inner").reset_index(drop=True)
    rad_cols = [c for c in rad.columns if c not in ("patient_id", "study_id")]
    y = df["label"].to_numpy()
    print(f"PI-CAI n={len(df)} csPCa+={y.sum()} | imaging broken on 50% of test, severity {args.severity}")

    methods = ["clinical", "static", "TransCal/CPCS", "evidential/TMC", "MoRA"]
    auc = {m: [] for m in methods}; ec = {m: [] for m in methods}
    sr_conf, sr_mora = [], []
    for seed in range(args.seeds):
        rng = np.random.RandomState(seed)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
        si = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]).fit(df[rad_cols].iloc[tr])
        sc = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]).fit(df[CLIN].iloc[tr])
        Xi_tr, Xi_te = si.transform(df[rad_cols].iloc[tr]), si.transform(df[rad_cols].iloc[te])
        Xc_tr, Xc_te = sc.transform(df[CLIN].iloc[tr]), sc.transform(df[CLIN].iloc[te])
        ytr, yte = y[tr], y[te]
        mi = LogisticRegression(max_iter=2000, C=0.5).fit(Xi_tr, ytr)
        mc = LogisticRegression(max_iter=2000, C=0.5).fit(Xc_tr, ytr)
        # break imaging on 50% of test
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
        sr_conf.append(selective_risk(conf / (w_global + EPS), err, 0.8))   # global down-weights by joint shift
        sr_mora.append(selective_risk(conf * np.minimum(ri, rc), err, 0.8))  # MoRA per-modality reliability

    def stat(d, k): return float(np.mean(d[k])), float(np.std(d[k]))
    print("\n=== AUROC under modality failure (mean±sd, %d seeds) ===" % args.seeds)
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
    json.dump({"severity": args.severity, "seeds": args.seeds, "lam": args.lam,
               "auroc": {m: round(stat(auc, m)[0], 4) for m in methods},
               "ece": {m: round(stat(ec, m)[0], 4) for m in methods},
               "mora_delta_winrate": {m: [round(mora_a.mean() - np.array(auc[m]).mean(), 4),
                                          round(float(np.mean(mora_a > np.array(auc[m]))), 3)]
                                      for m in ["static", "TransCal/CPCS", "evidential/TMC"]},
               "selective_risk_80": {"weighted_conformal": round(float(np.mean(sr_conf)), 4),
                                     "mora_deferral": round(float(np.mean(sr_mora)), 4)}},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(f"\nwrote {OUT}/summary.json")


if __name__ == "__main__":
    main()

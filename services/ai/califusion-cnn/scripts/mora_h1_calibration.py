#!/usr/bin/env python3
"""
scripts/mora_h1_calibration.py — H1: reliability-adaptive calibration under shift.

H1 (RESEARCH_DESIGN §3): calibration error (ECE/Brier/NLL) under modality-specific shift
is lower for reliability-adaptive temperature than for STATIC temperature (Guo'17) and
≤ importance-weighted calibration (TransCal/CPCS, Wang'20/Park'20).

Methods (all fit on SOURCE only; no target labels):
  uncalibrated   raw fused probability
  static_temp    single T minimising NLL on clean source-val (Guo 2017)
  transcal       importance-weighted T (CPCS/TransCal): domain-discriminator weights
                 source→target, weighted-NLL temperature (uses unlabeled target features)
  mora_global    T(x)=softplus(ψ·[1, 1-r_global]) — reliability-adaptive, single global signal
  mora_permod    T(x)=softplus(ψ·[1, 1-r_img, 1-r_clin]) — per-modality (the ablation vs global)
                 ψ fit on source-val under SIMULATED shift (NLL), per RESEARCH_DESIGN §2.3.

H1 verdict: mora_* ECE & NLL under shift < static_temp and ≤ transcal (paired over seeds).
Per-modality-vs-global is reported as an ablation (H2 already failed for prediction/deferral).

Run:  python scripts/mora_h1_calibration.py [--seeds 20] [--severity 2.5]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from califusion.data.clinical_unified import build_unified_preprocessor, UNIFIED_FEATURES
from califusion.mora.reliability import MahalanobisReliability

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "mora_h1")
COND = ["clean", "img_subset", "clin_subset"]
EPS = 1e-6


def _logit(p): return np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
def _sig(z): return 1.0 / (1.0 + np.exp(-z))


def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p): return float(np.mean((p - y) ** 2))


def fit_static_T(z, y):
    f = lambda T: nll(y, _sig(z / max(T, 1e-3)))
    return float(minimize_scalar(f, bounds=(0.05, 20), method="bounded").x)


def fit_weighted_T(z, y, w):
    w = w / (w.mean() + EPS)
    def f(T):
        p = np.clip(_sig(z / max(T, 1e-3)), EPS, 1 - EPS)
        return -np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(minimize_scalar(f, bounds=(0.05, 20), method="bounded").x)


def fit_mora_T(z, y, feats):
    """T(x)=softplus(ψ·feats)+0.25 ; fit ψ on (simulated-shifted) source-val NLL. feats: (n,k)."""
    k = feats.shape[1]
    def Tof(psi):
        return np.log1p(np.exp(np.clip(feats @ psi, -20, 20))) + 0.25
    def f(psi):
        p = np.clip(_sig(z / Tof(psi)), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 1e-3 * np.sum(psi ** 2)
    psi0 = np.zeros(k); psi0[0] = np.log(np.e - 1)  # softplus(0)~0.31 -> T~0.56 baseline; ok
    res = minimize(f, psi0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--severity", type=float, default=2.5)
    ap.add_argument("--shift_frac", type=float, default=0.5)
    args = ap.parse_args()

    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))
    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    df = rad.merge(clin.drop(columns=["label"]), on="PatientID", how="inner").reset_index(drop=True)
    y = df["label"].to_numpy(int)
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    print(f"cohort n={len(df)} | radiomics={len(rad_cols)} | prev={y.mean():.3f}")

    methods = ["uncalibrated", "static_temp", "transcal", "mora_global", "mora_permod"]
    rows = []
    for seed in range(args.seeds):
        rng = np.random.RandomState(3000 + seed)
        src, tgt = train_test_split(np.arange(len(y)), test_size=0.40, stratify=y, random_state=seed)
        fit_i, val_i = train_test_split(src, test_size=0.35, stratify=y[src], random_state=seed)
        img_pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        clin_pre = build_unified_preprocessor()
        Xi_f = img_pipe.fit_transform(df[rad_cols].iloc[fit_i]); Xc_f = clin_pre.fit_transform(df[UNIFIED_FEATURES].iloc[fit_i])
        base = LogisticRegression(max_iter=2000, C=0.5).fit(np.hstack([Xi_f, Xc_f]), y[fit_i])
        Rimg = MahalanobisReliability().fit(Xi_f, y[fit_i]); Rclin = MahalanobisReliability().fit(Xc_f, y[fit_i])
        Rglob = MahalanobisReliability().fit(np.hstack([Xi_f, Xc_f]), y[fit_i])
        img_std = Xi_f.std(0, keepdims=True); clin_std = Xc_f.std(0, keepdims=True)

        def transform(idx): return img_pipe.transform(df[rad_cols].iloc[idx]), clin_pre.transform(df[UNIFIED_FEATURES].iloc[idx])
        Xi_v, Xc_v = transform(val_i); Xi_t, Xc_t = transform(tgt)
        yv, yt = y[val_i], y[tgt]

        # base probs/logits
        zv = _logit(base.predict_proba(np.hstack([Xi_v, Xc_v]))[:, 1])
        # static temperature on clean val
        T_static = fit_static_T(zv, yv)
        # TransCal/CPCS: domain discriminator src(fit) vs tgt -> importance weights on val
        Dx = np.vstack([np.hstack([Xi_f, Xc_f]), np.hstack([Xi_t, Xc_t])])
        Dy = np.r_[np.zeros(len(fit_i)), np.ones(len(tgt))]
        disc = LogisticRegression(max_iter=1000, C=1.0).fit(Dx, Dy)
        dv = disc.predict_proba(np.hstack([Xi_v, Xc_v]))[:, 1]
        w_v = np.clip(dv / (1 - dv + EPS), 0.05, 20)
        T_transcal = fit_weighted_T(zv, yv, w_v)
        # MoRA: build simulated-shifted val set (mix clean/img/clin at varied severity), fit ψ
        sim_z, sim_y, sim_rimg, sim_rclin, sim_rglob = [], [], [], [], []
        for _ in range(3):
            Xi_s, Xc_s = Xi_v.copy(), Xc_v.copy()
            mode = rng.choice(["clean", "img", "clin"]); sev = rng.uniform(1.0, 3.0)
            mask = rng.random(len(val_i)) < 0.5
            if mode == "img": Xi_s[mask] += sev * rng.standard_normal(Xi_s[mask].shape) * img_std
            elif mode == "clin": Xc_s[mask] += sev * rng.standard_normal(Xc_s[mask].shape) * clin_std
            Zs = np.hstack([Xi_s, Xc_s])
            sim_z.append(_logit(base.predict_proba(Zs)[:, 1])); sim_y.append(yv)
            sim_rimg.append(Rimg.reliability(Xi_s)); sim_rclin.append(Rclin.reliability(Xc_s)); sim_rglob.append(Rglob.reliability(Zs))
        sz = np.concatenate(sim_z); sy = np.concatenate(sim_y)
        sri = np.concatenate(sim_rimg); src_ = np.concatenate(sim_rclin); srg = np.concatenate(sim_rglob)
        psi_g = fit_mora_T(sz, sy, np.c_[np.ones_like(srg), 1 - srg])
        psi_p = fit_mora_T(sz, sy, np.c_[np.ones_like(sri), 1 - sri, 1 - src_])

        def mora_T(feats, psi): return np.log1p(np.exp(np.clip(feats @ psi, -20, 20))) + 0.25

        for cond in COND:
            Xi, Xc = Xi_t.copy(), Xc_t.copy()
            if cond != "clean":
                mask = rng.random(len(tgt)) < args.shift_frac
                if cond == "img_subset": Xi[mask] += args.severity * rng.standard_normal(Xi[mask].shape) * img_std
                else: Xc[mask] += args.severity * rng.standard_normal(Xc[mask].shape) * clin_std
            Z = np.hstack([Xi, Xc]); zt = _logit(base.predict_proba(Z)[:, 1])
            ri = Rimg.reliability(Xi); rc = Rclin.reliability(Xc); rg = Rglob.reliability(Z)
            preds = {
                "uncalibrated": _sig(zt),
                "static_temp": _sig(zt / T_static),
                "transcal": _sig(zt / T_transcal),
                "mora_global": _sig(zt / mora_T(np.c_[np.ones_like(rg), 1 - rg], psi_g)),
                "mora_permod": _sig(zt / mora_T(np.c_[np.ones_like(ri), 1 - ri, 1 - rc], psi_p)),
            }
            for mth, p in preds.items():
                rows.append({"seed": seed, "cond": cond, "method": mth,
                             "ece": ece(yt, p), "brier": brier(yt, p), "nll": nll(yt, p)})

    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

    def cell(cond, mth, k): return float(R[(R.cond == cond) & (R.method == mth)][k].mean())
    table = {c: {m: {k: round(cell(c, m, k), 4) for k in ("ece", "brier", "nll")} for m in methods} for c in COND}

    # H1 verdict: under shift, mora_permod ECE & NLL < static_temp and <= transcal (paired win-rate)
    def winrate(cond, a, b, key, lower_better=True):
        pa = R[(R.cond == cond) & (R.method == a)].set_index("seed")[key]
        pb = R[(R.cond == cond) & (R.method == b)].set_index("seed")[key]
        d = (pa < pb) if lower_better else (pa > pb)
        return round(float(d.mean()), 3)

    shift = ["img_subset", "clin_subset"]
    h1 = {}
    for c in shift:
        h1[c] = {
            "permod_vs_static_ece_win": winrate(c, "mora_permod", "static_temp", "ece"),
            "permod_vs_static_nll_win": winrate(c, "mora_permod", "static_temp", "nll"),
            "permod_vs_transcal_ece_win": winrate(c, "mora_permod", "transcal", "ece"),
            "permod_vs_global_ece_win": winrate(c, "mora_permod", "mora_global", "ece"),
        }
    beats_static = all(h1[c]["permod_vs_static_ece_win"] >= 0.6 and h1[c]["permod_vs_static_nll_win"] >= 0.6 for c in shift)
    ge_transcal = all(h1[c]["permod_vs_transcal_ece_win"] >= 0.5 for c in shift)
    verdict = "SUPPORTED" if (beats_static and ge_transcal) else "NOT SUPPORTED"
    summary = {"n_seeds": args.seeds, "severity": args.severity, "table_ece_brier_nll": table,
               "H1_winrates": h1, "H1_verdict": verdict,
               "note": "per-modality vs global is the H2 ablation for calibration; report both."}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== H1: ECE under shift (lower=better) ===")
    print(f"  {'cond':11s} {'uncal':>7s} {'static':>7s} {'transcal':>8s} {'mora_glob':>9s} {'mora_perm':>9s}")
    for c in COND:
        t = table[c]
        print(f"  {c:11s} {t['uncalibrated']['ece']:7.3f} {t['static_temp']['ece']:7.3f} "
              f"{t['transcal']['ece']:8.3f} {t['mora_global']['ece']:9.3f} {t['mora_permod']['ece']:9.3f}")
    print("\n  NLL under shift:")
    for c in COND:
        t = table[c]
        print(f"  {c:11s} static {t['static_temp']['nll']:.3f} | transcal {t['transcal']['nll']:.3f} | "
              f"mora_perm {t['mora_permod']['nll']:.3f}")
    bar = "=" * 66
    print(f"\n{bar}")
    for c in shift:
        print(f"  {c}: MoRA-permod vs static ECE-win {h1[c]['permod_vs_static_ece_win']:.0%} / "
              f"NLL-win {h1[c]['permod_vs_static_nll_win']:.0%}; vs TransCal ECE-win "
              f"{h1[c]['permod_vs_transcal_ece_win']:.0%}; vs global ECE-win {h1[c]['permod_vs_global_ece_win']:.0%}")
    print(f"  H1 (reliability-adaptive calibration beats static & ≥ TransCal under shift): {verdict}")
    print(bar)


if __name__ == "__main__":
    main()

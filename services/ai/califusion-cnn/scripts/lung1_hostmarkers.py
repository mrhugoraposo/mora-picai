#!/usr/bin/env python3
"""
scripts/lung1_hostmarkers.py — #6: extra-tumoral CT "host" biomarkers for Lung1 2-yr OS.

Motivation (ADR-0008 + Option-3 result): the TUMOR ROI carries no signal complementary to
clinical staging (deep CNN, radiomics, AND foundation encoders all confirm redundancy). So
any real imaging contribution must come from OUTSIDE the tumor — host physiology that staging
cannot contain:
  - Emphysema / COPD severity from lung parenchyma density (%LAA-950/-910, PD15, lung HU stats).
  - Body-composition proxy (thoracic skeletal-muscle & fat area) — sarcopenia surrogate.
Both are validated independent prognostic factors in NSCLC and orthogonal to TNM stage.

Lung is segmented DIRECTLY from CT (threshold + remove border-connected external air) → no
RTSTRUCT dependency, orientation-free. Leakage-safe (intensity/geometry only; classifier per fold).

Outputs: data/processed/lung1_hostmarkers.csv + a clinical/host/fusion eval with ΔAUROC CI,
complementarity, and Gate-6. Run: python scripts/lung1_hostmarkers.py [--limit N] [--eval-only]
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")
from scipy import stats as sstats, ndimage
from skimage.segmentation import clear_border
from skimage.measure import label

from califusion.data import dicom_preprocess as D
from califusion.data import tcia_masks as TM
from califusion.data.clinical_unified import get_unified_xy_lung1, build_unified_preprocessor, UNIFIED_FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "NSCLC-Radiomics")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "hostmarkers")


def segment_lung(vol):
    """CT-direct lung mask: interior air (HU<-400) not connected to the image border."""
    air = vol < -400
    interior = np.stack([clear_border(air[z]) for z in range(vol.shape[0])])  # drop external air
    lbl = label(interior)
    if lbl.max() == 0:
        return np.zeros_like(air)
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    # keep components > 0.5% of volume (the two lungs; excludes trachea-only specks)
    keep = np.where(sizes > 0.005 * vol.size)[0]
    return np.isin(lbl, keep)


def emphysema_feats(vol, lung, vox_ml):
    hu = vol[lung]
    n = len(hu)
    p = np.percentile(hu, [15, 50])
    return {
        "lung_vol_ml": n * vox_ml,
        "LAA950": float(np.mean(hu < -950)), "LAA910": float(np.mean(hu < -910)),
        "LAA856": float(np.mean(hu < -856)),
        "lung_PD15": float(p[0]), "lung_median_hu": float(p[1]),
        "lung_mean_hu": float(hu.mean()), "lung_std_hu": float(hu.std()),
        "lung_skew": float(sstats.skew(hu)), "lung_kurtosis": float(sstats.kurtosis(hu)),
    }


def bodycomp_feats(vol, lung, spacing, vox_area_cm2):
    """Thoracic body-composition proxy at the mid-lung axial slice (no vertebral landmark —
    documented limitation). Muscle HU [-29,150], fat HU [-190,-30] within the body mask."""
    zs = np.where(lung.any((1, 2)))[0]
    if len(zs) == 0:
        return {}
    z = int(zs.mean())
    sl = vol[z]
    body = ndimage.binary_fill_holes(sl > -500)
    lbl = label(body);
    if lbl.max() == 0:
        return {}
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    body = lbl == sizes.argmax()                      # largest CC = patient body
    soft = body & ~lung[z]
    muscle = soft & (sl >= -29) & (sl <= 150)
    fat = soft & (sl >= -190) & (sl <= -30)
    body_a = float(body.sum() * vox_area_cm2)
    mus_a = float(muscle.sum() * vox_area_cm2); fat_a = float(fat.sum() * vox_area_cm2)
    return {"body_area_cm2": body_a, "muscle_area_cm2": mus_a, "fat_area_cm2": fat_a,
            "muscle_frac": mus_a / max(body_a, 1e-6), "fat_muscle_ratio": fat_a / max(mus_a, 1e-6)}


def extract_one(pdir):
    ct, _, _ = TM.find_series_dirs(pdir)
    if not ct:
        return None
    vol, zpos, spacing = D.load_ct_volume(ct)
    lung = segment_lung(vol)
    if lung.sum() < 5000:
        return None
    zsp = float(np.median(np.abs(np.diff(np.sort(zpos))))) or 3.0
    vox_ml = spacing[0] * spacing[1] * zsp / 1000.0
    vox_area = spacing[0] * spacing[1] / 100.0  # cm^2
    f = {"PatientID": os.path.basename(pdir)}
    f.update(emphysema_feats(vol, lung, vox_ml))
    f.update(bodycomp_feats(vol, lung, spacing, vox_area))
    return f


def build(limit=0):
    pdirs = sorted(d for d in glob.glob(os.path.join(RAW, "LUNG1-*")) if os.path.isdir(d))
    if limit:
        pdirs = pdirs[:limit]
    rows = []
    for i, p in enumerate(pdirs, 1):
        try:
            f = extract_one(p)
            if f:
                rows.append(f)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {repr(e)[:70]}")
        if i % 50 == 0:
            print(f"  ...{i}/{len(pdirs)} extracted={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(PROC, exist_ok=True)
    df.to_csv(os.path.join(PROC, "lung1_hostmarkers.csv"), index=False)
    print(f"extracted {len(df)} patients × {df.shape[1]-1} host markers")
    return df


def evaluate(df):
    import json
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    usable, Xc_raw, y, ids = get_unified_xy_lung1()
    clin = pd.DataFrame({"PatientID": ids, "label": y})
    for c in UNIFIED_FEATURES:
        clin[c] = Xc_raw[c].to_numpy()
    m = df.merge(clin, on="PatientID", how="inner").reset_index(drop=True)
    y = m["label"].to_numpy(int)
    host_cols = [c for c in df.columns if c != "PatientID"]
    print(f"\nmerged n={len(m)} | host markers={len(host_cols)} | prev={y.mean():.3f}")

    def num_pipe(cols, kind):
        clf = LogisticRegression(max_iter=2000, C=0.5) if kind == "logreg" else HistGradientBoostingClassifier(random_state=0)
        pre = ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]),
             [c for c in cols if c not in ("gender", "overall_stage", "histology")]),
            ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="UNK")),
                              ("o", __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"]).OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             [c for c in cols if c in ("gender", "overall_stage", "histology")]),
        ])
        return Pipeline([("p", pre), ("c", clf)])

    def oof(cols, kind, seed):
        X = m[cols]; o = np.full(len(y), np.nan)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            mm = clone(num_pipe(cols, kind)); mm.fit(X.iloc[tr], y[tr]); o[te] = mm.predict_proba(X.iloc[te])[:, 1]
        return o

    def cv(cols):
        best = None
        for kind in ("logreg", "gboost"):
            a = [roc_auc_score(y, oof(cols, kind, s)) for s in range(5)]
            if best is None or np.mean(a) > best[1]:
                best = (kind, float(np.mean(a)), float(np.std(a)), oof(cols, "logreg", 0))
        return best

    clin_cols = UNIFIED_FEATURES
    sets = {"clinical": clin_cols, "host_only": host_cols, "fusion": clin_cols + host_cols}
    res = {}
    print("\n=== #6 Host-marker eval (5×5 CV AUROC) ===")
    for name, cols in sets.items():
        k, mu, sd, _ = cv(cols); res[name] = (k, mu, sd)
        print(f"  {name:10s} {k:7s} AUROC {mu:.3f} ± {sd:.3f}")
    # paired bootstrap ΔAUROC fusion - clinical
    pc = oof(clin_cols, res["clinical"][0], 0); pf = oof(clin_cols + host_cols, res["fusion"][0], 0)
    rng = np.random.RandomState(0); deltas = []
    for _ in range(2000):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], pf[idx]) - roc_auc_score(y[idx], pc[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5]); dmean = res["fusion"][1] - res["clinical"][1]
    corr = float(np.corrcoef(pc, pf - pc)[0, 1])
    gate = (dmean >= 0.03) and (lo > 0)
    print(f"\n  ΔAUROC fusion−clinical = {dmean:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  host-only {res['host_only'][1]:.3f} | corr(clin, host-residual) {corr:.3f}")
    print(f"  GATE-6 (host markers add complementary signal, Δ≥0.03 & CI>0): {'PASS' if gate else 'FAIL'}")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"n": len(m), "auroc": {k: {"model": v[0], "mean": round(v[1], 4), "sd": round(v[2], 4)} for k, v in res.items()},
               "delta_fusion_clinical": round(dmean, 4), "delta_ci": [round(lo, 4), round(hi, 4)],
               "gate6_pass": bool(gate)}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    csv = os.path.join(PROC, "lung1_hostmarkers.csv")
    if args.eval_only and os.path.exists(csv):
        df = pd.read_csv(csv); print(f"loaded {len(df)} cached host-marker rows")
    else:
        df = build(args.limit)
    if len(df) >= 20 and not args.limit:
        evaluate(df)
    elif args.limit:
        print("(smoke run — skip eval)")


if __name__ == "__main__":
    main()

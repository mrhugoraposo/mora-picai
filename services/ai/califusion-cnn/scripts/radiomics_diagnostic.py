#!/usr/bin/env python3
"""
scripts/radiomics_diagnostic.py  —  Gate 1 diagnostic: does the GTV carry 2-yr-OS signal?

Extracts dependency-free radiomic features (first-order HU statistics + shape +
gradient-texture heterogeneity) from each patient's GTV mask region, then runs the same
patient-level 5x5 CV / classical models as the clinical baseline. Purpose: determine
whether the imaging *data* carries 2-yr-OS signal that the deep 2.5D encoder failed to
extract (Gate 1 FAIL diagnosis). Aerts et al. 2014 report C-index ~0.65 on Lung1 with
hand-crafted radiomics — this checks whether that signal is recoverable here.

Features are intensity/geometry only (no labels) -> leakage-safe; the classifier is fit
per-CV-fold. Raw HU is used (not lung-windowed) for proper first-order radiomics.

Outputs:
  data/processed/radiomics_features.csv     PatientID + features + label (cached; reused)
  results/radiomics_diagnostic/summary.json  per-model CV AUROC + verdict vs deep image arm

Run:  python scripts/radiomics_diagnostic.py [--limit N]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")
from scipy import stats as sstats

from califusion.data import dicom_preprocess as D
from califusion.data import tcia_masks as TM
from califusion.data.clinical_unified import get_unified_xy_lung1

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw", "NSCLC-Radiomics")
OUT = os.path.join(ROOT, "results", "radiomics_diagnostic")


def firstorder(hu):
    """First-order HU statistics over GTV voxels."""
    hu = hu.astype(np.float64)
    p = np.percentile(hu, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(hu, bins=32)
    prob = hist / max(1, hist.sum()); prob = prob[prob > 0]
    return {
        "fo_mean": hu.mean(), "fo_std": hu.std(), "fo_min": hu.min(), "fo_max": hu.max(),
        "fo_p10": p[0], "fo_p25": p[1], "fo_median": p[2], "fo_p75": p[3], "fo_p90": p[4],
        "fo_iqr": p[3] - p[1], "fo_range": hu.max() - hu.min(),
        "fo_skew": float(sstats.skew(hu)), "fo_kurtosis": float(sstats.kurtosis(hu)),
        "fo_energy": float(np.sum(hu ** 2) / len(hu)), "fo_rms": float(np.sqrt(np.mean(hu ** 2))),
        "fo_entropy": float(-np.sum(prob * np.log2(prob))),
        "fo_mad": float(np.mean(np.abs(hu - hu.mean()))),
    }


def shape(mask, spacing, zspacing):
    """Shape descriptors from the binary GTV mask."""
    zs, ys, xs = np.where(mask > 0)
    n = len(zs)
    vox_mm3 = spacing[0] * spacing[1] * zspacing
    vol = n * vox_mm3
    dz = (zs.max() - zs.min() + 1) * zspacing
    dy = (ys.max() - ys.min() + 1) * spacing[0]
    dx = (xs.max() - xs.min() + 1) * spacing[1]
    # surface proxy: boundary voxels (6-neighbour exposed faces)
    m = mask > 0
    faces = 0
    for ax in range(3):
        faces += np.sum(m & ~np.roll(m, 1, ax)) + np.sum(m & ~np.roll(m, -1, ax))
    surf = faces * np.mean([spacing[0] * zspacing, spacing[1] * zspacing, spacing[0] * spacing[1]])
    sphericity = (np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / max(surf, 1e-6)
    return {
        "sh_volume_mm3": vol, "sh_nvox": float(n),
        "sh_extent_z": dz, "sh_extent_y": dy, "sh_extent_x": dx,
        "sh_max_extent": max(dx, dy, dz), "sh_elongation": min(dx, dy, dz) / max(dx, dy, dz),
        "sh_surface": float(surf), "sh_sphericity": float(sphericity),
        "sh_surf_to_vol": float(surf / max(vol, 1e-6)),
    }


def texture(vol_hu, mask):
    """Gradient/Laplacian heterogeneity proxies within the GTV (texture surrogate)."""
    gz, gy, gx = np.gradient(vol_hu.astype(np.float64))
    gmag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
    m = mask > 0
    gm = gmag[m]
    # simple 3D Laplacian via 6-neighbour
    lap = (-6.0 * vol_hu
           + np.roll(vol_hu, 1, 0) + np.roll(vol_hu, -1, 0)
           + np.roll(vol_hu, 1, 1) + np.roll(vol_hu, -1, 1)
           + np.roll(vol_hu, 1, 2) + np.roll(vol_hu, -1, 2))
    return {
        "tx_grad_mean": float(gm.mean()), "tx_grad_std": float(gm.std()),
        "tx_grad_p90": float(np.percentile(gm, 90)),
        "tx_lap_var": float(lap[m].var()), "tx_lap_absmean": float(np.abs(lap[m]).mean()),
    }


def log_features(vol_hu, mask):
    """Laplacian-of-Gaussian multi-scale first-order within the GTV (where radiomic
    prognostic signal concentrates — Aerts' top features were filtered-image stats)."""
    from scipy import ndimage
    m = mask > 0
    out = {}
    for sigma in (1.0, 2.0, 3.0):
        fl = ndimage.gaussian_laplace(vol_hu.astype(np.float64), sigma=sigma)
        v = fl[m]
        hist, _ = np.histogram(v, bins=32); pr = hist / max(1, hist.sum()); pr = pr[pr > 0]
        out[f"log{int(sigma)}_mean"] = float(v.mean())
        out[f"log{int(sigma)}_std"] = float(v.std())
        out[f"log{int(sigma)}_energy"] = float(np.mean(v ** 2))
        out[f"log{int(sigma)}_entropy"] = float(-np.sum(pr * np.log2(pr)))
    return out


def _haralick(P):
    """Haralick features from a (normalised) gray-level co-occurrence matrix P[L,L]."""
    L = P.shape[0]
    i, j = np.mgrid[0:L, 0:L]
    px = P.sum(1); py = P.sum(0)
    mux = (i * P).sum(); muy = (j * P).sum()
    sx = np.sqrt(((i - mux) ** 2 * P).sum()); sy = np.sqrt(((j - muy) ** 2 * P).sum())
    pp = P[P > 0]
    corr = (((i * j * P).sum() - mux * muy) / (sx * sy)) if sx > 1e-9 and sy > 1e-9 else 0.0
    return {
        "glcm_contrast": float(((i - j) ** 2 * P).sum()),
        "glcm_dissimilarity": float((np.abs(i - j) * P).sum()),
        "glcm_homogeneity": float((P / (1.0 + (i - j) ** 2)).sum()),
        "glcm_energy": float((P ** 2).sum()),
        "glcm_entropy": float(-(pp * np.log2(pp)).sum()),
        "glcm_correlation": float(corr),
    }


def glcm_features(vol_hu, mask, levels=32, lo=-1000.0, hi=400.0):
    """Masked GLCM on the max-area GTV slice (HU discretised to `levels`; background
    co-occurrences removed). Averaged over 4 directions."""
    from skimage.feature import graycomatrix
    areas = mask.reshape(mask.shape[0], -1).sum(1)
    z = int(np.argmax(areas))
    m2 = mask[z] > 0
    ys, xs = np.where(m2)
    if len(ys) < 5:
        return {k: 0.0 for k in _haralick(np.eye(2) / 2)}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sl = np.clip(vol_hu[z, y0:y1, x0:x1], lo, hi)
    q = ((sl - lo) / (hi - lo) * (levels - 1)).astype(np.uint8) + 1   # levels 1..levels
    q[~(mask[z, y0:y1, x0:x1] > 0)] = 0                              # background = 0
    P = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=levels + 1, symmetric=True, normed=False).astype(np.float64)
    Pa = P[:, :, 0, :].sum(2)          # sum over directions
    Pa[0, :] = 0; Pa[:, 0] = 0         # drop background co-occurrences
    s = Pa.sum()
    if s < 1:
        return {k: 0.0 for k in _haralick(np.eye(2) / 2)}
    return _haralick(Pa[1:, 1:] / s)   # drop background level, renormalise


def extract_one(pdir):
    ct, rt, _ = TM.find_series_dirs(pdir)
    if not ct or not rt:
        return None
    vol, zpos, spacing = D.load_ct_volume(ct)
    mask, _ = TM.materialize_gtv_mask(ct, rt, vol)
    zspacing = float(np.median(np.abs(np.diff(np.sort(zpos))))) or 3.0
    hu = vol[mask > 0]
    if len(hu) < 10:
        return None
    feats = {}
    feats.update(firstorder(hu))
    feats.update(shape(mask, spacing, zspacing))
    feats.update(texture(vol, mask))
    feats.update(log_features(vol, mask))
    feats.update(glcm_features(vol, mask))
    return feats


def build_features(limit=0):
    cache_csv = os.path.join(PROC, "radiomics_features.csv")
    usable, _, y, ids = get_unified_xy_lung1()
    label = dict(zip(ids, y))
    pdirs = sorted(d for d in glob.glob(os.path.join(RAW, "LUNG1-*")) if os.path.isdir(d))
    if limit:
        pdirs = pdirs[:limit]
    rows = []
    for i, pdir in enumerate(pdirs, 1):
        pid = os.path.basename(pdir)
        if pid not in label:
            continue
        try:
            f = extract_one(pdir)
            if f is None:
                continue
            rows.append({"PatientID": pid, **f, "label": int(label[pid])})
        except Exception as e:
            print(f"  skip {pid}: {repr(e)[:80]}")
        if i % 50 == 0:
            print(f"  ...{i}/{len(pdirs)} extracted={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(PROC, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    print(f"extracted {len(df)} patients x {df.shape[1]-2} features -> {cache_csv}")
    return df


def cv_auroc(df):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    feat_cols = [c for c in df.columns if c not in ("PatientID", "label")]
    X = df[feat_cols].to_numpy(np.float64); y = df["label"].to_numpy(int)
    models = {
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "hist_gboost": Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("clf", HistGradientBoostingClassifier(random_state=0))]),
    }
    out = {}
    for name, mdl in models.items():
        aucs = []
        for rep in range(5):
            skf = StratifiedKFold(5, shuffle=True, random_state=rep)
            oof = np.full(len(y), np.nan)
            for tr, te in skf.split(X, y):
                from sklearn.base import clone
                m = clone(mdl); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y, oof))
        out[name] = (float(np.mean(aucs)), float(np.std(aucs)))
    return out, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cache_csv = os.path.join(PROC, "radiomics_features.csv")
    if os.path.exists(cache_csv) and not args.limit:
        df = pd.read_csv(cache_csv)
        print(f"loaded cached features: {len(df)} patients")
    else:
        df = build_features(args.limit)
    if len(df) < 20:
        print("too few patients extracted; aborting CV"); return
    res, feat_cols = cv_auroc(df)
    os.makedirs(OUT, exist_ok=True)
    best = max(res.values(), key=lambda t: t[0])[0]
    summary = {
        "n_patients": len(df), "n_features": len(feat_cols),
        "cv": "5x5 repeated stratified", "auroc": {k: {"mean": round(v[0], 4), "sd": round(v[1], 4)} for k, v in res.items()},
        "best_radiomics_auroc": round(best, 4),
        "deep_image_arm_auroc": 0.415, "clinical_arm_auroc": 0.582,
        "gate1_threshold": 0.63,
        "verdict": ("imaging signal EXISTS (radiomics >> deep arm) -> deep encoder is the bottleneck"
                    if best >= 0.58 else
                    "imaging signal WEAK even for radiomics -> data-limited, not just encoder"),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== RADIOMICS DIAGNOSTIC ===")
    for k, (m, s) in res.items():
        print(f"  {k:14s} AUROC {m:.3f} ± {s:.3f}")
    print(f"  deep image arm: 0.415 | clinical arm: 0.582 | gate: 0.63")
    print(f"  VERDICT: {summary['verdict']}")


if __name__ == "__main__":
    main()

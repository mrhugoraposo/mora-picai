#!/usr/bin/env python3
"""
scripts/picai_radiomics.py — prostate-MRI radiomics for PI-CAI csPCa (P1, ADR-0009).

Per-case multi-sequence radiomics from the whole-gland ROI. MRI differs from CT:
  - No absolute intensity scale → z-score normalize each sequence WITHIN the gland.
  - Multiple sequences (T2W anatomy, ADC diffusion [csPCa = low ADC], HBV high-b DWI
    [csPCa = high signal]) → extract per sequence and concatenate.
  - Sequences have different geometries → resample ADC/HBV/mask onto the T2W grid.

Whole-gland masks: picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/<pid>_<sid>.nii.gz.
Leakage-safe: intensity/geometry features only (no labels, no fitting). Classifier fit per fold later.

Output: data/processed/picai_radiomics_features.csv (patient_id, study_id, <features>).
Run:  python scripts/picai_radiomics.py [--limit N]
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats as sstats

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "data", "raw", "PI-CAI", "images")
MASKS = os.path.join(ROOT, "data", "raw", "PI-CAI", "picai_labels",
                     "anatomical_delineations", "whole_gland", "AI", "Bosma22b")
PROC = os.path.join(ROOT, "data", "processed")
SEQS = ["t2w", "adc", "hbv"]


def resample_to(ref, moving, is_mask=False):
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(moving)


def firstorder(v, prefix):
    v = v.astype(np.float64)
    p = np.percentile(v, [10, 25, 50, 75, 90])
    hist, _ = np.histogram(v, bins=32); pr = hist / max(1, hist.sum()); pr = pr[pr > 0]
    return {
        f"{prefix}_mean": v.mean(), f"{prefix}_std": v.std(),
        f"{prefix}_p10": p[0], f"{prefix}_median": p[2], f"{prefix}_p90": p[4],
        f"{prefix}_iqr": p[3] - p[1], f"{prefix}_skew": float(sstats.skew(v)),
        f"{prefix}_kurtosis": float(sstats.kurtosis(v)), f"{prefix}_energy": float(np.mean(v ** 2)),
        f"{prefix}_entropy": float(-np.sum(pr * np.log2(pr))),
        f"{prefix}_min": v.min(), f"{prefix}_max": v.max(),
    }


def texture(arr3d, mask3d, prefix):
    gz, gy, gx = np.gradient(arr3d.astype(np.float64))
    gm = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)[mask3d]
    return {f"{prefix}_grad_mean": float(gm.mean()), f"{prefix}_grad_std": float(gm.std()),
            f"{prefix}_grad_p90": float(np.percentile(gm, 90))}


def shape_feats(mask_img):
    m = sitk.GetArrayFromImage(mask_img) > 0
    sp = mask_img.GetSpacing()  # (x,y,z)
    vox = sp[0] * sp[1] * sp[2]
    n = int(m.sum()); vol = n * vox
    faces = 0
    for ax in range(3):
        faces += np.sum(m & ~np.roll(m, 1, ax)) + np.sum(m & ~np.roll(m, -1, ax))
    surf = faces * np.mean([sp[0] * sp[1], sp[0] * sp[2], sp[1] * sp[2]])
    sph = (np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / max(surf, 1e-6)
    zs, ys, xs = np.where(m)
    return {"sh_volume_mm3": vol, "sh_nvox": float(n), "sh_surface": float(surf),
            "sh_sphericity": float(sph), "sh_surf_to_vol": float(surf / max(vol, 1e-6)),
            "sh_extent_z": float((zs.max() - zs.min() + 1) * sp[2])}


def find_case_files(case_dir):
    """Return {seq: path} for a case dir (files like <pid>_<sid>_t2w.mha)."""
    out = {}
    for seq in SEQS:
        hits = glob.glob(os.path.join(case_dir, f"*_{seq}.mha"))
        if hits:
            out[seq] = hits[0]
    return out


def extract_case(case_dir):
    files = find_case_files(case_dir)
    if "t2w" not in files:
        return None
    base = os.path.basename(files["t2w"]).replace("_t2w.mha", "")  # <pid>_<sid>
    mask_path = os.path.join(MASKS, base + ".nii.gz")
    if not os.path.exists(mask_path):
        return None
    t2w = sitk.ReadImage(files["t2w"], sitk.sitkFloat32)
    mask = resample_to(t2w, sitk.ReadImage(mask_path), is_mask=True)
    m = sitk.GetArrayFromImage(mask) > 0
    if m.sum() < 50:
        return None
    feats = {"patient_id": int(base.split("_")[0]), "study_id": int(base.split("_")[1])}
    feats.update(shape_feats(mask))
    for seq in SEQS:
        if seq not in files:
            continue
        img = sitk.ReadImage(files[seq], sitk.sitkFloat32)
        if seq != "t2w":
            img = resample_to(t2w, img)
        arr = sitk.GetArrayFromImage(img)
        gland = arr[m]
        mu, sd = gland.mean(), gland.std() + 1e-6
        arr_z = (arr - mu) / sd                      # z-score within gland (MRI normalization)
        feats.update(firstorder(arr_z[m], seq))
        feats.update(texture(arr_z, m, seq))
    return feats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    cases = sorted(d for d in glob.glob(os.path.join(IMG, "*")) if os.path.isdir(d))
    if args.limit:
        cases = cases[:args.limit]
    print(f"found {len(cases)} case dirs under {IMG}")
    rows = []
    for i, c in enumerate(cases, 1):
        try:
            f = extract_case(c)
            if f:
                rows.append(f)
        except Exception as e:
            print(f"  skip {os.path.basename(c)}: {repr(e)[:80]}")
        if i % 50 == 0:
            print(f"  ...{i}/{len(cases)} extracted={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(PROC, exist_ok=True)
    out = os.path.join(PROC, "picai_radiomics_features.csv")
    df.to_csv(out, index=False)
    print(f"extracted {len(df)} cases × {df.shape[1]-2} features -> {out}")
    if len(df):
        print("sequences present per case (sample):",
              {s: int(df.filter(like=f'{s}_').shape[1] > 0) for s in SEQS})


if __name__ == "__main__":
    main()

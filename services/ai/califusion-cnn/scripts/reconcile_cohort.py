#!/usr/bin/env python3
"""
scripts/reconcile_cohort.py  —  Phase B reconciliation: imaging ∩ clinical.

Joins the preprocessed image cache (data/processed/image_cache.npz) with the unified
Lung1 clinical labels/features (clinical_unified.get_unified_xy_lung1) to form the
matched multimodal cohort used by training (Phase C).

Outputs (data/processed/):
  clinical_unified.csv   PatientID, age, gender, overall_stage, histology, label  (matched; NaN preserved — AUTHORITATIVE)
  clinical.npz           ids, y, age(float NaN), gender/overall_stage/histology (U; missing->'UNK'), feature_names  (convenience)
  cohort_manifest.json   n_imaging, n_clinical_usable, n_matched, prevalence, unmatched lists

Leakage note: features are RAW (no fitting here). The unified preprocessor is fit per
training split downstream (build_unified_preprocessor).

Guard: if matched N ≪ 400 the script prints a STOP-AND-ASK banner (Current-Priority
pause condition) and exits non-zero so the caller surfaces it.

Run:  python scripts/reconcile_cohort.py
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

from califusion.data.clinical_unified import get_unified_xy_lung1, UNIFIED_FEATURES

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
MIN_MATCHED = 400  # Current-Priority pause threshold (~400 usable expected)


def main():
    cache_path = os.path.join(PROC, "image_cache.npz")
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found — run scripts/preprocess.py first.")
        sys.exit(2)
    z = np.load(cache_path)
    img_ids = set(z.files)
    print(f"imaging-cached patients: {len(img_ids)}")

    usable, X, y, ids = get_unified_xy_lung1()
    clin = pd.DataFrame({"PatientID": ids})
    for c in UNIFIED_FEATURES:
        clin[c] = X[c].to_numpy()
    clin["label"] = y
    clin_ids = set(clin["PatientID"])
    print(f"clinical usable patients (2-yr OS labelled): {len(clin_ids)} (prev {y.mean():.3f})")

    matched_ids = sorted(img_ids & clin_ids)
    matched = clin[clin["PatientID"].isin(matched_ids)].reset_index(drop=True)
    matched = matched.sort_values("PatientID").reset_index(drop=True)
    n = len(matched); n_pos = int(matched["label"].sum())
    print(f"\n=== MATCHED multimodal cohort: n={n} | pos={n_pos} | neg={n-n_pos} | "
          f"prevalence={n_pos/max(1,n):.3f} ===")

    img_only = sorted(img_ids - clin_ids)
    clin_only = sorted(clin_ids - img_ids)
    print(f"imaging-only (no clinical label): {len(img_only)}{(' e.g. '+str(img_only[:5])) if img_only else ''}")
    print(f"clinical-only (no image cache):   {len(clin_only)}{(' e.g. '+str(clin_only[:5])) if clin_only else ''}")

    # authoritative CSV (NaN preserved)
    matched.to_csv(os.path.join(PROC, "clinical_unified.csv"), index=False)
    # convenience npz (no pickle): age float w/ NaN; categoricals U w/ missing->UNK
    np.savez(
        os.path.join(PROC, "clinical.npz"),
        ids=np.array(matched["PatientID"], dtype="U24"),
        y=matched["label"].to_numpy(np.int8),
        age=matched["age"].to_numpy(np.float32),
        gender=matched["gender"].fillna("UNK").to_numpy(dtype="U16"),
        overall_stage=matched["overall_stage"].fillna("UNK").to_numpy(dtype="U8"),
        histology=matched["histology"].fillna("UNK").to_numpy(dtype="U16"),
        feature_names=np.array(UNIFIED_FEATURES, dtype="U16"),
    )

    manifest = {
        "n_imaging_cached": len(img_ids),
        "n_clinical_usable": len(clin_ids),
        "n_matched": n, "n_pos": n_pos, "n_neg": n - n_pos,
        "prevalence_matched": round(n_pos / max(1, n), 4),
        "n_imaging_only": len(img_only), "n_clinical_only": len(clin_only),
        "imaging_only_ids": img_only, "clinical_only_ids": clin_only,
        "unified_features": UNIFIED_FEATURES,
        "artifacts": ["clinical_unified.csv", "clinical.npz", "image_cache.npz"],
    }
    with open(os.path.join(PROC, "cohort_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote clinical_unified.csv, clinical.npz, cohort_manifest.json -> {os.path.relpath(PROC, ROOT)}")

    if n < MIN_MATCHED:
        print("\n" + "=" * 70)
        print(f"  STOP-AND-ASK: matched N={n} < {MIN_MATCHED} (expected ~420).")
        print("  Per Current Priority, surface this before committing to training.")
        print("  Likely causes: incomplete download, preprocess failures, ID mismatch.")
        print("=" * 70)
        sys.exit(1)
    print(f"\nOK: matched cohort N={n} ≥ {MIN_MATCHED}. Ready for Phase C.")


if __name__ == "__main__":
    main()

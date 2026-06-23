"""
califusion.data.multimodal  —  leakage-safe matched-cohort loader + per-split prep.

Phase C training consumes this. The unified clinical preprocessor is fit on the TRAIN
split ONLY (never val/test); the image cache is keyed by PatientID. This replaces the
earlier train_multimodal.py path that loaded a globally pre-transformed clinical matrix
(which would leak scaler/one-hot statistics across the split boundary).

Contract:
  load_matched_cohort() -> (ids, X_raw_df[UNIFIED_FEATURES], y, image_cache)
  prepare_split(...)     -> per-split (ids, X_transformed, y) + clinical_dim + fitted prep
                            (prep fit on TRAIN only; calibrators/thresholds fit on VAL later)
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

from .clinical_unified import build_unified_preprocessor, UNIFIED_FEATURES
from .datasets import patient_level_split

PROC_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed")


def load_matched_cohort(proc_dir: str = PROC_DEFAULT):
    """Load the matched multimodal cohort produced by reconcile_cohort.py.

    Returns ids (np[str]), X_raw (DataFrame of UNIFIED_FEATURES, NaN preserved),
    y (np[int]), image_cache (dict PatientID -> (K,H,W) float32). Every returned id is
    guaranteed to have a cache entry; rows without one are dropped with a warning count.
    """
    csv = os.path.join(proc_dir, "clinical_unified.csv")
    npz = os.path.join(proc_dir, "image_cache.npz")
    if not os.path.exists(csv) or not os.path.exists(npz):
        raise FileNotFoundError(f"need {csv} and {npz} (run preprocess.py + reconcile_cohort.py)")
    df = pd.read_csv(csv)
    cache_npz = np.load(npz)
    cached = set(cache_npz.files)
    keep = df["PatientID"].isin(cached)
    dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    if dropped:
        print(f"[multimodal] dropped {dropped} clinical rows with no image cache entry")
    ids = df["PatientID"].to_numpy().astype(str)
    X_raw = df[UNIFIED_FEATURES].copy()
    y = df["label"].to_numpy().astype(int)
    cache = {pid: cache_npz[pid].astype(np.float32) for pid in ids}
    return ids, X_raw, y, cache


def prepare_split(ids, X_raw, y, seed: int = 0, ratios=(0.70, 0.15, 0.15)):
    """Patient-level split + TRAIN-only unified preprocessor fit.

    Returns dict with keys train/val/test -> (ids, X_transformed[np.float32], y), plus
    'clinical_dim' and 'preprocessor' (the fitted ColumnTransformer). No leakage: the
    preprocessor sees only training rows; val/test are transform-only.
    """
    tr, va, te = patient_level_split(ids, y, ratios=ratios, seed=seed)
    prep = build_unified_preprocessor()
    prep.fit(X_raw.iloc[tr])                       # TRAIN ONLY
    out = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        Xt = prep.transform(X_raw.iloc[idx]).astype(np.float32)
        out[name] = (ids[idx], Xt, y[idx])
    out["clinical_dim"] = out["train"][1].shape[1]
    out["preprocessor"] = prep
    out["split_idx"] = {"train": tr, "val": va, "test": te}
    return out

#!/usr/bin/env python3
"""
build_demo.py — build a SELF-CONTAINED, SYNTHETIC demonstration for the MoRA console.

Generates NON-IDENTIFIABLE synthetic data (NOT real patients; NO PI-CAI data is used or
required) with a fixed seed, fits lightweight per-modality demo models + label-free
reliability references, and writes the artifacts the console expects into ./demo_assets/.
The console falls back to these assets when no real deployment models are present, so the
full MoRA workflow — per-modality prediction, reliability, reliability-gated fusion,
selective deferral, and modality attribution — can be demonstrated out of the box.

Everything written here is synthetic and exists only to illustrate the mechanism; none of
it reproduces or contains any patient-derived value.

Usage:
    python build_demo.py
    ./run_local.sh            # then open the console and replay the DEMO-### cases
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 20260623
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_assets")

# Illustrative (synthetic) feature schema — names mirror the real pipeline's shape only.
IMG_COLS = ["firstorder_mean", "firstorder_entropy", "glcm_contrast", "glcm_correlation",
            "shape_sphericity", "glrlm_run_length", "log_sigma3_mean", "wavelet_energy"]
CLIN_COLS = ["psa", "psa_density", "prostate_volume_ml", "age"]
LAMBDA = 2.0


def synth(n, rng, pos_rate=0.45):
    """Synthetic, non-identifiable cases. Positives get a mild signal shift in both arms."""
    y = (rng.rand(n) < pos_rate).astype(int)
    Xi = rng.normal(0, 1, (n, len(IMG_COLS))) + y[:, None] * 0.8
    Xc = np.column_stack([
        np.clip(rng.normal(6, 3, n) + y * 4.0, 0.2, None),          # psa
        np.clip(rng.normal(0.12, 0.05, n) + y * 0.08, 0.01, None),  # psa density
        np.clip(rng.normal(45, 14, n), 15, None),                   # prostate volume (ml)
        np.clip(rng.normal(66, 7, n), 40, 90),                      # age
    ])
    return Xi, Xc, y


def _d(X_src_scaled, x_scaled):
    """Domain discriminator d = P(this case is 'target') — mirrors mora_engine._reliability."""
    X = np.vstack([X_src_scaled, np.atleast_2d(x_scaled)])
    dom = np.r_[np.zeros(len(X_src_scaled)), np.ones(1)]
    disc = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, C=0.5))]).fit(X, dom)
    return float(disc.predict_proba(np.atleast_2d(x_scaled))[0, 1])


def d_ref_of(X_src_scaled, rng, k=25):
    """Calibrate d_ref = median in-distribution d (hold out a source point vs the rest)."""
    idx = rng.choice(len(X_src_scaled), size=min(k, len(X_src_scaled)), replace=False)
    ds = [_d(np.delete(X_src_scaled, i, axis=0), X_src_scaled[i]) for i in idx]
    return float(np.median(ds))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)
    Xi_s, Xc_s, y_s = synth(160, rng)
    Xi_r, Xc_r, y_r = synth(24, rng, pos_rate=0.5)

    # Fit on DataFrames (with column names) so the console's DataFrame .transform() is warning-free.
    Xi_s_df = pd.DataFrame(Xi_s, columns=IMG_COLS)
    Xc_s_df = pd.DataFrame(Xc_s, columns=CLIN_COLS)
    img_scaler = Pipeline([("s", StandardScaler())]).fit(Xi_s_df)
    img_model = LogisticRegression(max_iter=1000).fit(img_scaler.transform(Xi_s_df), y_s)
    clin_scaler = Pipeline([("s", StandardScaler())]).fit(Xc_s_df)
    clin_model = LogisticRegression(max_iter=1000).fit(clin_scaler.transform(Xc_s_df), y_s)

    Xi_s_sc = img_scaler.transform(Xi_s_df).astype(np.float32)
    Xc_s_sc = clin_scaler.transform(Xc_s_df).astype(np.float32)
    img_dref = d_ref_of(Xi_s_sc, rng)
    clin_dref = d_ref_of(Xc_s_sc, rng)

    joblib.dump({"feature_cols": IMG_COLS, "scaler": img_scaler, "model": img_model},
                os.path.join(OUT, "picai_imaging.joblib"))
    joblib.dump({"feature_cols": CLIN_COLS, "scaler": clin_scaler, "model": clin_model},
                os.path.join(OUT, "picai_clinical.joblib"))
    joblib.dump({"operating_threshold": 0.5, "clinical_feature_cols": CLIN_COLS,
                 "clinical": {"X_src_scaled": Xc_s_sc, "d_ref": clin_dref, "lambda": LAMBDA},
                 "imaging": {"X_src_scaled": Xi_s_sc, "d_ref": img_dref, "lambda": LAMBDA}},
                os.path.join(OUT, "picai_reliability.joblib"))

    rep = pd.DataFrame(Xi_r, columns=IMG_COLS)
    for j, c in enumerate(CLIN_COLS):
        rep[c] = Xc_r[:, j]
    rep.insert(0, "patient_id", [f"DEMO-{i:03d}" for i in range(len(rep))])
    rep.insert(1, "center", rng.choice(["SYN-CENTER-A", "SYN-CENTER-B"], len(rep)))
    rep["label"] = y_r
    rep.to_csv(os.path.join(OUT, "picai_replay.csv"), index=False)

    json.dump({"synthetic": True,
               "datasets": {"picai": {"label": "csPCa (SYNTHETIC DEMO — not real patients)",
                                      "n": int(len(rep)), "n_replay": int(len(rep)),
                                      "note": "All values are randomly generated for UI "
                                              "demonstration only; no patient data is used."}}},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"demo assets -> {OUT}")
    print(f"  imaging d_ref={img_dref:.3f}  clinical d_ref={clin_dref:.3f}  (lambda={LAMBDA})")
    print(f"  {len(rep)} synthetic replay cases (DEMO-000 .. DEMO-{len(rep)-1:03d})")


if __name__ == "__main__":
    main()

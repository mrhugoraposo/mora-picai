"""
Phase B regression tests — pure-logic units (no network, no DICOM required).

Covers the geometry of the 2.5D pipeline and the unified-clinical coarse mappers,
plus the orientation-guard decision rule. Run:  python -m pytest tests/test_phase_b.py
(or: python tests/test_phase_b.py for a dependency-free smoke run).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from califusion.data import dicom_preprocess as D
from califusion.data.clinical_unified import (
    _coarse_stage_lung1, _coarse_histology, _gender,
    UNIFIED_FEATURES, lung1_to_unified,
)
import pandas as pd


# ---------- 2.5D geometry ----------

def test_apply_window_maps_to_unit_interval():
    vol = np.array([[-2000.0, 5000.0], [-600.0, 150.0]], dtype=np.float32)
    w = D.apply_window(vol, level=-600, width=1500)
    assert w.min() >= 0.0 and w.max() <= 1.0
    # value at the window centre maps to 0.5
    assert abs(float(D.apply_window(np.array([[-600.0]]), -600, 1500)[0, 0]) - 0.5) < 1e-6


def test_crop_roi_shape_and_padding():
    sl = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    # centre crop
    p = D.crop_roi(sl, (50, 50), size=96)
    assert p.shape == (96, 96)
    # near-border crop is padded, never wrong-shaped
    p2 = D.crop_roi(sl, (1, 1), size=96)
    assert p2.shape == (96, 96)


def test_make_2p5d_channels_and_centroid():
    Z, H, W = 20, 128, 128
    vol = np.zeros((Z, H, W), np.float32)
    mask = np.zeros((Z, H, W), np.uint8)
    mask[10, 64, 64] = 1
    mask[9:12, 60:68, 60:68] = 1
    zc, (yc, xc) = D.gtv_centroid(mask)
    assert zc == 10 and 60 <= yc <= 67 and 60 <= xc <= 67
    t = D.make_2p5d(vol, (zc, yc, xc), k=3, roi=96)
    assert t.shape == (3, 96, 96) and t.dtype == np.float32


def test_gtv_centroid_empty_raises():
    try:
        D.gtv_centroid(np.zeros((4, 4, 4), np.uint8))
        assert False, "expected ValueError on empty mask"
    except ValueError:
        pass


def test_orientation_guard_rule():
    """The guard keeps whichever orientation reads as tissue (HU near 0), not air."""
    Z, H, W = 12, 32, 32
    vol = np.full((Z, H, W), -900.0, np.float32)   # lung background
    vol[2] = 30.0                                   # 'tumour' tissue near top of stack
    mask = np.zeros((Z, H, W), np.uint8); mask[2, 10:20, 10:20] = 1
    hu_asis = vol[np.where(mask > 0)].mean()
    hu_flip = vol[np.where(mask[::-1] > 0)].mean()
    # as-is sits on tissue (~30), flip sits on air (~-900) -> rule must NOT flip
    flipped = abs(hu_asis - 0.0) > abs(hu_flip - 0.0)
    assert not flipped and hu_asis > -300 and hu_flip < -300


# ---------- unified clinical coarse mappers ----------

def test_coarse_stage():
    assert _coarse_stage_lung1("IIIa") == "III"
    assert _coarse_stage_lung1("IIIb") == "III"
    assert _coarse_stage_lung1("II") == "II"
    assert _coarse_stage_lung1("I") == "I"
    assert _coarse_stage_lung1("IV") == "IV"
    assert _coarse_stage_lung1(np.nan) != _coarse_stage_lung1(np.nan) or pd.isna(_coarse_stage_lung1(np.nan))


def test_coarse_histology():
    assert _coarse_histology("Adenocarcinoma") == "adenocarcinoma"
    assert _coarse_histology("squamous cell carcinoma") == "squamous"
    assert _coarse_histology("large cell") == "large_cell"
    assert _coarse_histology("nos") == "nos"
    assert pd.isna(_coarse_histology(np.nan))


def test_gender_normalises():
    assert _gender("male") == "male" and _gender("F") == "female"
    assert pd.isna(_gender(np.nan))


def test_lung1_to_unified_columns():
    df = pd.DataFrame({
        "age": [60, np.nan], "gender": ["male", "female"],
        "Overall.Stage": ["IIIb", "I"], "Histology": ["squamous cell carcinoma", np.nan],
    })
    out = lung1_to_unified(df)
    assert list(out.columns) == UNIFIED_FEATURES
    assert out.loc[0, "overall_stage"] == "III" and out.loc[0, "histology"] == "squamous"
    assert pd.isna(out.loc[1, "histology"])


# ---------- leakage-safe multimodal split ----------

def _synth_cohort(n=120, seed=0):
    rng = np.random.RandomState(seed)
    ids = np.array([f"P{i:03d}" for i in range(n)])
    X = pd.DataFrame({
        "age": rng.normal(65, 9, n),
        "gender": rng.choice(["male", "female"], n),
        "overall_stage": rng.choice(["I", "II", "III", "IV"], n),
        "histology": rng.choice(["adenocarcinoma", "squamous", "large_cell", "nos"], n),
    })
    y = rng.randint(0, 2, n)
    return ids, X, y


def test_prepare_split_no_patient_overlap():
    from califusion.data.multimodal import prepare_split
    ids, X, y = _synth_cohort()
    s = prepare_split(ids, X, y, seed=1)
    tr, va, te = (set(s[k][0]) for k in ("train", "val", "test"))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr | va | te) == len(ids)


def test_prepare_split_preprocessor_fit_on_train_only():
    """The scaler mean must equal the TRAIN age (median-imputed) mean, not the global mean."""
    from califusion.data.multimodal import prepare_split
    ids, X, y = _synth_cohort()
    s = prepare_split(ids, X, y, seed=2)
    tr_idx = s["split_idx"]["train"]
    prep = s["preprocessor"]
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    train_age = X.iloc[tr_idx]["age"]
    med = train_age.median()
    expected_mean = float(train_age.fillna(med).mean())
    # first numeric column is age; mean_ aligns to imputed-then-scaled input
    assert abs(float(scaler.mean_[0]) - expected_mean) < 1e-6
    # and it must DIFFER from the global mean (else fit leaked across the split)
    assert abs(float(scaler.mean_[0]) - float(X["age"].mean())) > 1e-9 or len(tr_idx) == len(ids)


def test_prepare_split_consistent_dim_and_unknown_category():
    from califusion.data.multimodal import prepare_split
    ids, X, y = _synth_cohort()
    # inject a category that may land only in val/test -> handle_unknown='ignore' must not error
    X.loc[0, "histology"] = "rare_subtype"
    s = prepare_split(ids, X, y, seed=3)
    d = s["clinical_dim"]
    for k in ("train", "val", "test"):
        assert s[k][1].shape[1] == d and s[k][1].dtype == np.float32


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn(); passed += 1; print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} Phase B unit tests passed")

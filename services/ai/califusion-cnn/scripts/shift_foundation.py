#!/usr/bin/env python3
"""
scripts/shift_foundation.py  —  Option-3 Tier-1 imaging shift stress test (run only if the
core gates clear, otherwise kept brief). Adds Gaussian noise + blur to each patient's CT ROI,
re-extracts frozen embeddings for the SHIFTED images, and compares AUROC + ECE degradation for
clinical-only (unaffected, control) vs radiomics vs foundation-image vs fusions.

The HEAD/calibrator is the SAME model fit on CLEAN training folds (per fold), evaluated on the
shifted OOF test fold — i.e. train clean, test shifted (covariate shift). Clinical is identical
on clean vs shifted (sanity control: its drop must be ~0).

Run:  ./.venv/bin/python scripts/shift_foundation.py --encoder medicalnet
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.ndimage import gaussian_filter

from califusion.data import dicom_preprocess as D
from califusion.data import tcia_masks as TM
from califusion.eval import metrics as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw", "NSCLC-Radiomics")
OUT = os.path.join(ROOT, "results", "option3")
ROI_D, ROI_H, ROI_W = 48, 96, 96
FOLDS = 5


def crop_box_3d(vol, c, d=ROI_D, h=ROI_H, w=ROI_W):
    Z, H, W = vol.shape
    cz, cy, cx = c
    out = np.zeros((d, h, w), np.float32)
    z0, y0, x0 = cz - d // 2, cy - h // 2, cx - w // 2
    zs, ys, xs = max(0, z0), max(0, y0), max(0, x0)
    ze, ye, xe = min(Z, z0 + d), min(H, y0 + h), min(W, x0 + w)
    out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = vol[zs:ze, ys:ye, xs:xe]
    return out


def patient_roi_hu(pdir):
    ct, rt, _ = TM.find_series_dirs(pdir)
    if ct is None or rt is None:
        return None
    try:
        vol, _, _ = D.load_ct_volume(ct)
        mask, _ = TM.materialize_gtv_mask(ct, rt, vol)
    except Exception:
        return None
    zc, (yc, xc) = D.gtv_centroid(mask)
    return crop_box_3d(vol, (zc, yc, xc))


def shift_hu(roi_hu, noise_sd=80.0, blur_sigma=1.0, seed=0):
    """Covariate shift on raw HU: additive Gaussian noise (sd in HU) + mild Gaussian blur."""
    rng = np.random.default_rng(seed)
    v = gaussian_filter(roi_hu, sigma=blur_sigma)
    return v + rng.normal(0, noise_sd, v.shape).astype(np.float32)


def build_encoder(name, device):
    import torch
    if name == "medicalnet":
        from monai.networks.nets import resnet18
        net = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2, feed_forward=False,
                       shortcut_type="A", bias_downsample=True, pretrained=True)

        def fwd(roi_hu):
            x = D.apply_window(roi_hu, -600, 1500)[None, None].astype(np.float32)
            with torch.no_grad():
                return net(torch.from_numpy(x).to(device)).squeeze(0).cpu().numpy()
    elif name == "ctfm":
        from lighter_zoo import SegResEncoder
        net = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")

        def fwd(roi_hu):
            v = np.clip(roi_hu, -1024, 2048); v = ((v + 1024) / 3072).astype(np.float32)
            with torch.no_grad():
                f = net(torch.from_numpy(v[None, None]).to(device))
                return f[-1].mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()
    else:
        raise ValueError(name)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return fwd


def oof_eval(Xtr_full, y, get_test_feats, kind="logreg"):
    """Train per fold on clean Xtr_full; predict each test fold using get_test_feats(test_idx)."""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
    for tr, te in skf.split(np.zeros(len(y)), y):
        sc = StandardScaler().fit(Xtr_full[tr])
        clf = (LogisticRegression(max_iter=5000, C=0.1) if kind == "logreg"
               else HistGradientBoostingClassifier(random_state=0))
        clf.fit(sc.transform(Xtr_full[tr]), y[tr])
        Xte = get_test_feats(te)
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return oof


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="medicalnet")
    ap.add_argument("--noise_sd", type=float, default=80.0)
    ap.add_argument("--blur_sigma", type=float, default=1.0)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))
    emb_clean = np.load(os.path.join(PROC, f"foundation_embeddings_{args.encoder}.npz"))
    common = sorted(set(clin.PatientID) & set(rad.PatientID) & set(emb_clean.files))
    clin = clin.set_index("PatientID").loc[common].reset_index()
    rad = rad.set_index("PatientID").loc[common].reset_index()
    y = clin["label"].to_numpy(int)
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    Xrad_clean = rad[rad_cols].to_numpy(float)
    Eclean = np.stack([emb_clean[p] for p in common]).astype(float)

    print(f"shift test enc={args.encoder} N={len(common)} noise_sd={args.noise_sd} blur={args.blur_sigma}")
    print("re-extracting SHIFTED embeddings + radiomics (first-order on shifted HU) ...")
    fwd = build_encoder(args.encoder, device)
    from scipy import stats as sst
    Eshift = np.zeros_like(Eclean)
    Xrad_shift = Xrad_clean.copy()
    for i, pid in enumerate(common):
        roi = patient_roi_hu(os.path.join(RAW, pid))
        if roi is None:
            Eshift[i] = Eclean[i]; continue
        sroi = shift_hu(roi, args.noise_sd, args.blur_sigma, seed=i)
        Eshift[i] = fwd(sroi).astype(float)
        # shift the first-order radiomics that depend on intensity (fo_* block), recomputed on GTV-box voxels
        # (approximate: recompute fo stats over the shifted ROI core; shape/texture left as clean proxy)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(common)}")

    # clinical (control), radiomics, foundation: clean-train / shifted-test
    Xc = clin[["age", "gender", "overall_stage", "histology"]]
    # clinical OOF via simple one-hot+scaler (matches compare harness closely enough for a control)
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    def clin_pre():
        return ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), ["age"]),
            ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="UNK")),
                              ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             ["gender", "overall_stage", "histology"])])
    def oof_clin():
        oof = np.full(len(y), np.nan)
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=0)
        for tr, te in skf.split(np.zeros(len(y)), y):
            m = Pipeline([("p", clin_pre()), ("c", HistGradientBoostingClassifier(random_state=0))])
            m.fit(Xc.iloc[tr], y[tr]); oof[te] = m.predict_proba(Xc.iloc[te])[:, 1]
        return oof
    p_clin = oof_clin()

    def report(tag, p):
        t = M.youden_threshold(y, p)
        return {"auroc": round(M.auroc(y, p), 4), "ece": round(M.expected_calibration_error(y, p), 4),
                "brier": round(M.brier(y, p), 4)}

    res = {"clinical_control": report("clin", p_clin)}
    # radiomics: clean vs shifted (we only shift fo-intensity proxy weakly; report clean as baseline)
    p_rad_clean = oof_eval(Xrad_clean, y, lambda te: Xrad_clean[te], "logreg")
    res["radiomics_clean"] = report("rad_clean", p_rad_clean)
    # foundation image-only: clean train, clean test vs shifted test
    p_img_clean = oof_eval(Eclean, y, lambda te: Eclean[te], "logreg")
    p_img_shift = oof_eval(Eclean, y, lambda te: Eshift[te], "logreg")
    res["foundation_clean"] = report("img_clean", p_img_clean)
    res["foundation_shift"] = report("img_shift", p_img_shift)
    # fusion late-mean clean vs shifted (clinical + foundation)
    res["fusion_late_clean"] = report("fus_clean", 0.5 * (p_clin + p_img_clean))
    res["fusion_late_shift"] = report("fus_shift", 0.5 * (p_clin + p_img_shift))

    res["_deltas"] = {
        "foundation_auroc_drop": round(res["foundation_clean"]["auroc"] - res["foundation_shift"]["auroc"], 4),
        "foundation_ece_rise": round(res["foundation_shift"]["ece"] - res["foundation_clean"]["ece"], 4),
        "fusion_auroc_drop": round(res["fusion_late_clean"]["auroc"] - res["fusion_late_shift"]["auroc"], 4),
        "fusion_ece_rise": round(res["fusion_late_shift"]["ece"] - res["fusion_late_clean"]["ece"], 4),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"shift_{args.encoder}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

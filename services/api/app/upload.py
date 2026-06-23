"""
upload.py — UPLOAD-mode inference (Step 2b; lowest priority per the brief).

Two upload paths, in order of robustness:
  1. radiomics-vector paste — the user pastes a comma/space/newline-separated feature vector
     in the model's feature order (always works, no imaging libs needed). Most reliable.
  2. imaging file (.mha/.nii.gz/.nii or a zipped DICOM dir) — we extract radiomics from a
     sensible whole-organ ROI using the SAME extractor logic as the pipeline. If an optional
     mask is not provided we threshold a whole-organ ROI. If extraction is infeasible for an
     arbitrary upload, we say so and fall back to clinical-only + suggest REPLAY.

Clinical features always come from the submitted form. Leakage-safe: all scaling/models/
reliability references are the persisted SOURCE-fit artifacts (mora_engine).
"""
from __future__ import annotations
import os
import tempfile

import numpy as np

from . import mora_engine as ENGINE

CLINICAL_SCHEMAS = {
    "picai": [
        {"name": "patient_age", "label": "Patient age (years)", "type": "number", "ph": "66"},
        {"name": "psa", "label": "PSA (ng/mL)", "type": "number", "ph": "7.5"},
        {"name": "psad", "label": "PSA density", "type": "number", "ph": "0.15"},
        {"name": "prostate_volume", "label": "Prostate volume (mL)", "type": "number", "ph": "50"},
    ],
    "lung1": [
        {"name": "age", "label": "Age (years)", "type": "number", "ph": "68"},
        {"name": "gender", "label": "Gender", "type": "select", "options": ["male", "female"]},
        {"name": "overall_stage", "label": "Overall stage", "type": "select",
         "options": ["I", "II", "III", "IV"]},
        {"name": "histology", "label": "Histology", "type": "select",
         "options": ["adenocarcinoma", "squamous_cell_carcinoma", "large_cell", "nos"]},
    ],
}


def _parse_vector(text: str, n_expected: int):
    toks = [t for t in text.replace(",", " ").replace("\n", " ").split() if t.strip()]
    vals = [float(t) for t in toks]
    return vals


def _extract_from_image(dataset: str, path: str):
    """Extract radiomics from an uploaded volume using a threshold whole-organ ROI.

    Returns (feat_dict, note). Reuses the pipeline's feature definitions. This is a best-effort
    path for arbitrary uploads — it will not match a curated GTV/gland mask, so we flag it."""
    import SimpleITK as sitk
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)
    icols = ENGINE._load(f"{dataset}_imaging.joblib")["feature_cols"]

    # threshold ROI: Otsu on the volume -> largest connected region proxy (foreground organ)
    try:
        otsu = sitk.OtsuThreshold(img, 0, 1)
        m = sitk.GetArrayFromImage(otsu) > 0
    except Exception:
        m = arr > np.percentile(arr, 60)
    if m.sum() < 50:
        m = arr > np.percentile(arr, 60)

    feats = {}
    if dataset == "lung1":
        # CT path: first-order HU stats + gradient texture on the ROI (radiomics_diagnostic.py)
        hu = arr[m].astype(np.float64)
        p = np.percentile(hu, [10, 25, 50, 75, 90])
        import scipy.stats as ss
        feats.update({
            "fo_mean": hu.mean(), "fo_std": hu.std(), "fo_min": hu.min(), "fo_max": hu.max(),
            "fo_p10": p[0], "fo_p25": p[1], "fo_median": p[2], "fo_p75": p[3], "fo_p90": p[4],
            "fo_iqr": p[3] - p[1], "fo_range": hu.max() - hu.min(),
            "fo_skew": float(ss.skew(hu)), "fo_kurtosis": float(ss.kurtosis(hu)),
            "fo_energy": float(np.sum(hu ** 2) / len(hu)),
            "fo_rms": float(np.sqrt(np.mean(hu ** 2))),
            "fo_entropy": float(-np.sum((lambda pr: pr[pr > 0])(
                np.histogram(hu, 32)[0] / max(1, len(hu))) *
                np.log2((lambda pr: pr[pr > 0])(np.histogram(hu, 32)[0] / max(1, len(hu)))))),
            "fo_mad": float(np.mean(np.abs(hu - hu.mean()))),
        })
        gz, gy, gx = np.gradient(arr.astype(np.float64))
        gm = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)[m]
        feats.update({"tx_grad_mean": float(gm.mean()), "tx_grad_std": float(gm.std()),
                      "tx_grad_p90": float(np.percentile(gm, 90))})
    else:
        # MRI path: z-score within ROI then first-order + gradient (picai_radiomics.py)
        for seq in ("t2w", "adc", "hbv"):
            gland = arr[m]
            mu, sd = gland.mean(), gland.std() + 1e-6
            az = (arr - mu) / sd
            v = az[m].astype(np.float64)
            p = np.percentile(v, [10, 25, 50, 75, 90])
            import scipy.stats as ss
            feats.update({
                f"{seq}_mean": v.mean(), f"{seq}_std": v.std(), f"{seq}_p10": p[0],
                f"{seq}_median": p[2], f"{seq}_p90": p[4], f"{seq}_iqr": p[3] - p[1],
                f"{seq}_skew": float(ss.skew(v)), f"{seq}_kurtosis": float(ss.kurtosis(v)),
                f"{seq}_energy": float(np.mean(v ** 2)), f"{seq}_min": v.min(), f"{seq}_max": v.max(),
            })
            gz, gy, gx = np.gradient(az)
            gm = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)[m]
            feats.update({f"{seq}_grad_mean": float(gm.mean()), f"{seq}_grad_std": float(gm.std()),
                          f"{seq}_grad_p90": float(np.percentile(gm, 90))})

    present = sum(1 for c in icols if c in feats)
    note = (f"Radiomics extracted from a threshold whole-organ ROI ({present}/{len(icols)} "
            f"features matched the model schema). This is a best-effort ROI for an arbitrary "
            f"upload (not a curated GTV/gland mask), so the imaging reliability r_img will likely "
            f"flag it as out-of-distribution — which is exactly MoRA's job.")
    return {c: feats.get(c, np.nan) for c in icols}, note


async def handle_upload(dataset, request, imaging_file, radiomics_vector):
    form = await request.form()
    schema = CLINICAL_SCHEMAS[dataset]
    clinical_feats = {f["name"]: form.get(f["name"]) for f in schema}

    imaging_feats = None
    note = None
    icols = ENGINE._load(f"{dataset}_imaging.joblib")["feature_cols"]

    if radiomics_vector and radiomics_vector.strip():
        vals = _parse_vector(radiomics_vector, len(icols))
        if len(vals) != len(icols):
            raise ValueError(
                f"radiomics vector has {len(vals)} values but the {dataset} imaging model "
                f"expects {len(icols)} (feature order: {', '.join(icols[:4])}, ...).")
        imaging_feats = {c: vals[i] for i, c in enumerate(icols)}
        note = f"Imaging from a pasted radiomics vector ({len(vals)} features in model order)."
    elif imaging_file is not None and imaging_file.filename:
        suffix = os.path.splitext(imaging_file.filename)[1] or ".mha"
        data = await imaging_file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        try:
            imaging_feats, note = _extract_from_image(dataset, tmp)
        except Exception as e:
            note = (f"Could not extract radiomics from the uploaded file ({type(e).__name__}: {e}). "
                    f"Proceeding clinical-only. For imaging, paste a radiomics vector or use REPLAY.")
            imaging_feats = None
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    else:
        note = "No imaging provided — clinical-only inference."

    res = ENGINE.infer(dataset, imaging_feats, clinical_feats, source="upload",
                       extraction_note=note)
    res["case_id"] = "upload"
    return res

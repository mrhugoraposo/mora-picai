#!/usr/bin/env python3
"""
scripts/preprocess.py  —  Phase B CT preprocessing -> cached 2.5D tensors.

For each patient under data/raw/<collection>/<PatientID>/:
  CT series + RTSTRUCT(GTV-1) -> HU -> lung window -> GTV-centroid ROI(roi) ->
  2.5D K-slice stack -> (K, roi, roi) float32 in [0,1].

Outputs (data/processed/):
  image_cache.npz        PatientID -> (K,roi,roi) float32   (consumed by datasets.Lung1Multimodal)
  preprocess_manifest.csv per-patient QC + diagnostics (status, gtv voxels, z-flip, alignment)
  preprocess_meta.json   config used + seeds + counts (reproducibility sidecar)
  ../results/preprocess_qc/<pid>_qc.png   sampled QC montages

No data fitting here (geometry only) -> leakage-safe and deterministic. Resumable:
patients already present in an existing image_cache.npz are skipped unless --force.

Run:  python scripts/preprocess.py [--limit N] [--qc-every 50] [--force]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")

from califusion.data import dicom_preprocess as D
from califusion.data import tcia_masks as TM

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
# CT series with documented skipped/irregular slices (data/README_DATA.md) — processed
# but flagged so downstream can audit/exclude; not silently dropped.
KNOWN_IRREGULAR = {"LUNG1-014", "LUNG1-021", "LUNG1-085"}


def load_config():
    cfg_path = os.path.join(ROOT, "configs", "default.yaml")
    try:
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {"preprocess": {"window_level": -600, "window_width": 1500,
                               "roi": 96, "k_slices": 3},
                "data": {"collection": "NSCLC-Radiomics"}}


def slice_uniformity(ct_dir: str):
    """Return (n_slices, max_gap/median_gap) to detect missing-slice irregularity."""
    try:
        _, zpos, _ = D.load_ct_volume(ct_dir)
        d = np.diff(np.sort(zpos))
        if len(d) == 0:
            return len(zpos), float("nan")
        med = float(np.median(np.abs(d)))
        return len(zpos), (float(np.max(np.abs(d))) / med if med else float("nan"))
    except Exception:
        return 0, float("nan")


def process_patient(patient_dir, pid, pp):
    ct_dir, rt_path, seg_path = TM.find_series_dirs(patient_dir)
    if not ct_dir or not rt_path:
        return None, {"patient_id": pid, "status": "missing_ct_or_rtstruct",
                      "ct": bool(ct_dir), "rtstruct": bool(rt_path)}
    vol, zpos, spacing = D.load_ct_volume(ct_dir)
    mask, info = TM.materialize_gtv_mask(ct_dir, rt_path, vol)
    tens = D.preprocess_patient(
        ct_dir, mask_volume=mask,
        level=pp["window_level"], width=pp["window_width"],
        k=pp["k_slices"], roi=pp["roi"])
    zc, (yc, xc) = D.gtv_centroid(mask)
    n_sl, gap_ratio = len(zpos), float("nan")
    d = np.diff(np.sort(zpos))
    if len(d):
        med = float(np.median(np.abs(d)))
        gap_ratio = round(float(np.max(np.abs(d))) / med, 3) if med else float("nan")
    rec = {
        "patient_id": pid, "status": "ok",
        "n_slices": n_sl, "gap_ratio": gap_ratio,
        "irregular_flag": pid in KNOWN_IRREGULAR or (gap_ratio == gap_ratio and gap_ratio > 1.5),
        "gtv_roi": info["gtv_roi"], "gtv_voxels": info["gtv_voxels"],
        "centroid_z": zc, "centroid_y": yc, "centroid_x": xc,
        "z_flipped": info["z_flipped"], "alignment_ok": info["alignment_ok"],
        "tensor_shape": "x".join(map(str, tens.shape)),
    }
    return tens.astype(np.float32), rec


def save_qc(pid, patient_dir, pp, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ct_dir, rt_path, _ = TM.find_series_dirs(patient_dir)
    vol, _, _ = D.load_ct_volume(ct_dir)
    mask, _ = TM.materialize_gtv_mask(ct_dir, rt_path, vol)
    vw = D.apply_window(vol, pp["window_level"], pp["window_width"])
    tens = D.preprocess_patient(ct_dir, mask_volume=mask, level=pp["window_level"],
                                width=pp["window_width"], k=pp["k_slices"], roi=pp["roi"])
    zc, _ = D.gtv_centroid(mask)
    k = tens.shape[0]
    fig, ax = plt.subplots(1, k + 1, figsize=(3.4 * (k + 1), 3.4))
    ax[0].imshow(vw[zc], cmap="gray"); ax[0].contour(mask[zc], colors="r", linewidths=0.8)
    ax[0].set_title(f"{pid} z={zc}\nCT+GTV")
    for i in range(k):
        ax[i + 1].imshow(tens[i], cmap="gray"); ax[i + 1].set_title(f"2.5D ch{i}")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N patients (0=all)")
    ap.add_argument("--qc-every", type=int, default=50, help="save a QC montage every N ok patients")
    ap.add_argument("--force", action="store_true", help="reprocess even if already cached")
    args = ap.parse_args()

    cfg = load_config()
    pp = cfg["preprocess"]
    collection = cfg["data"]["collection"]
    raw_dir = os.path.join(ROOT, "data", "raw", collection)
    out_dir = os.path.join(ROOT, "data", "processed")
    qc_dir = os.path.join(ROOT, "results", "preprocess_qc")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(qc_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "image_cache.npz")
    manifest_path = os.path.join(out_dir, "preprocess_manifest.csv")

    existing = {}
    if os.path.exists(cache_path) and not args.force:
        z = np.load(cache_path)
        existing = {k: z[k] for k in z.files}
        print(f"resuming: {len(existing)} patients already cached")

    patients = sorted(d for d in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(d))
    if args.limit:
        patients = patients[:args.limit]
    print(f"found {len(patients)} patient dirs under {raw_dir}")

    cache = dict(existing)
    records = []
    n_ok = n_fail = n_skip = 0
    for pdir in patients:
        pid = os.path.basename(pdir)
        if pid in cache and not args.force:
            n_skip += 1
            continue
        try:
            tens, rec = process_patient(pdir, pid, pp)
            if tens is None:
                records.append(rec); n_fail += 1
                print(f"  SKIP {pid}: {rec['status']}")
                continue
            cache[pid] = tens
            records.append(rec)
            n_ok += 1
            flag = " [FLIP]" if rec["z_flipped"] else ""
            flag += " [IRREG]" if rec["irregular_flag"] else ""
            if n_ok % max(1, args.qc_every) == 0:
                try:
                    save_qc(pid, pdir, pp, os.path.join(qc_dir, f"{pid}_qc.png"))
                except Exception:
                    pass
            print(f"  OK   {pid}  gtv={rec['gtv_voxels']:>6}  z={rec['centroid_z']}{flag}")
        except Exception as e:
            records.append({"patient_id": pid, "status": "error", "error": repr(e)[:160]})
            n_fail += 1
            print(f"  FAIL {pid}: {repr(e)[:120]}")

    # persist cache + manifest + meta sidecar
    if cache:
        np.savez_compressed(cache_path, **cache)
    import csv
    cols = ["patient_id", "status", "n_slices", "gap_ratio", "irregular_flag", "gtv_roi",
            "gtv_voxels", "centroid_z", "centroid_y", "centroid_x", "z_flipped",
            "alignment_ok", "tensor_shape", "ct", "rtstruct", "error"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in records:
            w.writerow({c: r.get(c, "") for c in cols})
    meta = {"collection": collection, "preprocess": pp,
            "n_cached_total": len(cache), "n_ok_this_run": n_ok,
            "n_fail_this_run": n_fail, "n_skipped_existing": n_skip,
            "n_z_flipped": sum(1 for r in records if r.get("z_flipped")),
            "n_irregular": sum(1 for r in records if r.get("irregular_flag")),
            "cache_path": os.path.relpath(cache_path, ROOT)}
    with open(os.path.join(out_dir, "preprocess_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDONE  ok={n_ok} fail={n_fail} skipped={n_skip}  total_cached={len(cache)}")
    print(f"cache    -> {os.path.relpath(cache_path, ROOT)}")
    print(f"manifest -> {os.path.relpath(manifest_path, ROOT)}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

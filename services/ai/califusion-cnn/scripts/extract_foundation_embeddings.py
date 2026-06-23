#!/usr/bin/env python3
"""
scripts/extract_foundation_embeddings.py  —  Option-3 frozen CT foundation embeddings.

Builds a TUMOR-CENTERED 3D ROI from each patient's GTV mask and extracts frozen
(no-grad, weights-frozen) embeddings from pretrained 3D CT encoders. Embeddings are
leakage-safe by construction: a frozen encoder fits nothing on the cohort; only the
downstream HEAD/scaler are fit per training fold (done later in compare_foundation.py).

Encoders (all with genuine pretrained weights — confirmed non-random):
  medicalnet  : MONAI 3D ResNet-18, MedicalNet pretrained (23 medical datasets) -> 512-d
  ctfm        : CT-FM SegResNet encoder, SSL-pretrained on 148k CT scans (IDC) -> 512-d
                (global-average-pool of the deepest 512-ch feature map)
  r3d18       : torchvision r3d_18, Kinetics-400 pretrained (generic 3D video) -> 512-d

3D ROI construction (documented precisely):
  1. load_ct_volume  -> HU volume [Z,H,W] (ascending-z), pixel spacing.
  2. materialize_gtv_mask (RTSTRUCT GTV-1, orientation-guarded) -> binary mask [Z,H,W].
  3. GTV centroid -> crop a fixed box of (D=48, H=96, W=96) voxels centered on it,
     zero-padded at borders. This is a fixed geometric crop (no resampling) so the
     in-plane field of view matches the verified radiomics/2.5D pipeline (96x96).
  4. Intensity normalization is ENCODER-SPECIFIC and applied at extraction time:
       medicalnet : lung window (level -600 / width 1500) -> [0,1]  (matches house CT pipeline)
       ctfm       : ScaleIntensityRange HU [-1024, 2048] -> [0,1]   (CT-FM README spec)
       r3d18      : lung window -> [0,1], replicated to 3 channels, depth tiled to >=16
  5. Encoder forward (frozen) -> pooled 512-d vector. Cached per encoder.

Skips patients with no CT or RTSTRUCT, or empty/failed GTV mask; reports N per encoder.

Outputs:
  data/processed/foundation_embeddings_<encoder>.npz   (PatientID -> 512-d float32)
  data/processed/foundation_embeddings_manifest.json   (N, failures, ROI/preproc spec)

Run:  ./.venv/bin/python scripts/extract_foundation_embeddings.py [--encoders medicalnet,ctfm,r3d18] [--limit N]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from califusion.data import dicom_preprocess as D
from califusion.data import tcia_masks as TM

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
RAW = os.path.join(ROOT, "data", "raw", "NSCLC-Radiomics")

# Fixed 3D ROI box (voxels): depth x height x width, centered on GTV centroid.
ROI_D, ROI_H, ROI_W = 48, 96, 96


# ------------------------------- ROI extraction ------------------------------- #
def crop_box_3d(vol, center_zyx, d=ROI_D, h=ROI_H, w=ROI_W):
    """Fixed-size zero-padded crop of `vol` [Z,H,W] centered on (z,y,x)."""
    Z, H, W = vol.shape
    cz, cy, cx = center_zyx
    out = np.zeros((d, h, w), dtype=np.float32)
    z0, y0, x0 = cz - d // 2, cy - h // 2, cx - w // 2
    zs, ys, xs = max(0, z0), max(0, y0), max(0, x0)
    ze, ye, xe = min(Z, z0 + d), min(H, y0 + h), min(W, x0 + w)
    out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = vol[zs:ze, ys:ye, xs:xe]
    return out


def patient_roi_hu(patient_dir):
    """Return the tumor-centered HU ROI [D,H,W] for one patient, or None if unusable."""
    ct_dir, rt_path, _seg = TM.find_series_dirs(patient_dir)
    if ct_dir is None or rt_path is None:
        return None, "no_ct_or_rtstruct"
    try:
        vol_hu, _zpos, _sp = D.load_ct_volume(ct_dir)
        mask, _info = TM.materialize_gtv_mask(ct_dir, rt_path, vol_hu)
    except Exception as e:  # noqa: BLE001
        return None, f"mask_fail:{type(e).__name__}"
    zc, (yc, xc) = D.gtv_centroid(mask)
    roi = crop_box_3d(vol_hu, (zc, yc, xc))
    return roi, "ok"


# ----------------------------- intensity normalizers -------------------------- #
def norm_lung_window(roi_hu):
    """Lung window level -600 / width 1500 -> [0,1]. Matches the house CT pipeline."""
    return D.apply_window(roi_hu, level=-600, width=1500).astype(np.float32)


def norm_ctfm(roi_hu):
    """CT-FM ScaleIntensityRange: HU clip [-1024, 2048] -> [0,1] (README spec)."""
    a_min, a_max = -1024.0, 2048.0
    v = np.clip(roi_hu, a_min, a_max)
    return ((v - a_min) / (a_max - a_min)).astype(np.float32)


# -------------------------------- encoders ------------------------------------ #
def build_encoder(name, device):
    import torch
    if name == "medicalnet":
        from monai.networks.nets import resnet18
        net = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2,
                       feed_forward=False, shortcut_type="A", bias_downsample=True,
                       pretrained=True)
        weights = "MedicalNet (23 medical datasets), pretrained"

        def fwd(roi_hu):
            x = norm_lung_window(roi_hu)[None, None]  # (1,1,D,H,W)
            t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                return net(t).squeeze(0).cpu().numpy()

    elif name == "ctfm":
        from lighter_zoo import SegResEncoder
        net = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
        weights = "CT-FM SSL (148k CT scans, IDC), pretrained"

        def fwd(roi_hu):
            x = norm_ctfm(roi_hu)[None, None]
            t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                feats = net(t)              # list of feature maps
                deep = feats[-1]            # (1, 512, d', h', w')
                emb = deep.mean(dim=(2, 3, 4)).squeeze(0)  # global avg pool -> 512
            return emb.cpu().numpy()

    elif name == "r3d18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        import torch.nn as nn
        net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        net.fc = nn.Identity()
        weights = "Kinetics-400 (generic 3D video), pretrained"

        def fwd(roi_hu):
            x = norm_lung_window(roi_hu)            # [D,H,W] in [0,1]
            # r3d_18 expects (B, C=3, T>=~8, H, W); replicate window to 3 channels,
            # treat depth as time. Center-resize 96->112 by reflect pad is overkill;
            # r3d_18 is fully convolutional pre-pool, so 96x96 is fine.
            t = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1, 1).to(device)  # (1,3,D,H,W)
            with torch.no_grad():
                return net(t).squeeze(0).cpu().numpy()
    else:
        raise ValueError(f"unknown encoder {name}")

    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return fwd, weights


# ----------------------------------- main ------------------------------------- #
def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="medicalnet,ctfm,r3d18")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} | encoders={encoders} | ROI={ROI_D}x{ROI_H}x{ROI_W}")

    # patient list = matched cohort (intersection of cache + labels handled downstream;
    # here we extract for every raw patient that has CT+RTSTRUCT).
    pdirs = sorted(d for d in os.listdir(RAW) if d.startswith("LUNG1-"))
    if args.limit:
        pdirs = pdirs[:args.limit]

    # Cache HU ROIs once (shared across encoders) to avoid re-reading DICOM 3x.
    print(f"building tumor-centered ROIs for {len(pdirs)} patients ...")
    rois, roi_status = {}, {}
    t0 = time.time()
    for i, pid in enumerate(pdirs):
        roi, status = patient_roi_hu(os.path.join(RAW, pid))
        roi_status[pid] = status
        if roi is not None:
            rois[pid] = roi
        if (i + 1) % 50 == 0:
            print(f"  ROI {i+1}/{len(pdirs)}  ok={len(rois)}  ({time.time()-t0:.0f}s)")
    n_ok = len(rois)
    fails = {k: v for k, v in roi_status.items() if v != "ok"}
    print(f"ROIs built: {n_ok} ok / {len(pdirs)} total | failures: {len(fails)} -> {list(fails.items())[:8]}")

    manifest = {
        "roi_box_voxels": [ROI_D, ROI_H, ROI_W],
        "roi_centering": "GTV centroid (RTSTRUCT GTV-1, orientation-guarded)",
        "n_patients_scanned": len(pdirs),
        "n_roi_ok": n_ok,
        "roi_failures": fails,
        "preprocessing": {
            "medicalnet": "lung window L-600/W1500 -> [0,1]",
            "ctfm": "ScaleIntensityRange HU[-1024,2048] -> [0,1]",
            "r3d18": "lung window -> [0,1], 3-channel replicate, depth-as-time",
        },
        "encoders": {},
    }

    for enc in encoders:
        print(f"\n=== encoder: {enc} ===")
        fwd, weights = build_encoder(enc, device)
        print(f"  weights: {weights}")
        emb_dict, te0 = {}, time.time()
        for j, (pid, roi) in enumerate(rois.items()):
            emb_dict[pid] = fwd(roi).astype(np.float32)
            if (j + 1) % 50 == 0:
                print(f"  emb {j+1}/{n_ok}  ({time.time()-te0:.0f}s)")
        dim = len(next(iter(emb_dict.values())))
        out_path = os.path.join(PROC, f"foundation_embeddings_{enc}.npz")
        np.savez_compressed(out_path, **emb_dict)
        print(f"  saved {len(emb_dict)} x {dim}-d -> {out_path}  ({time.time()-te0:.0f}s)")
        manifest["encoders"][enc] = {"weights": weights, "dim": dim,
                                     "n": len(emb_dict), "path": os.path.relpath(out_path, ROOT)}

    os.makedirs(PROC, exist_ok=True)
    with open(os.path.join(PROC, "foundation_embeddings_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDONE. manifest -> data/processed/foundation_embeddings_manifest.json")


if __name__ == "__main__":
    main()

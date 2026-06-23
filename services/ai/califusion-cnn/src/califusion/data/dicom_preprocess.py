"""
califusion.data.dicom_preprocess  —  CT preprocessing for Lung1.

Pipeline:
  1. Load DICOM CT series (pydicom) -> 3-D volume, apply RescaleSlope/Intercept -> HU.
  2. Lung window (default level -600, width 1500) -> clip -> [0,1].
  3. Resample to isotropic-ish spacing (default 1x1x3 mm) [optional, SimpleITK].
  4. Locate tumour from RTSTRUCT/SEG (GTV) -> centroid slice; crop fixed ROI box.
  5. Build 2.5D input: K adjacent axial slices around the GTV centroid stacked
     as channels (default K=3) -> (K, H, W) float32 in [0,1].

Leakage prevention: preprocessing parameters are FIXED (no fitting on data); any
patient-derived normalisation is per-volume only.

Dependencies: pydicom, numpy; optional SimpleITK (resampling),
pydicom-seg / rt-utils (mask parsing). Install via requirements.txt.
"""
from __future__ import annotations
import glob
import os
import numpy as np

try:
    import pydicom
    _HAS_PYDICOM = True
except Exception:
    _HAS_PYDICOM = False


LUNG_LEVEL = -600
LUNG_WIDTH = 1500
ROI = 96          # square crop side in pixels (post-window)
K_SLICES = 3      # 2.5D depth


def load_ct_volume(series_dir: str):
    """Return (volume_HU [Z,H,W], slice_z_positions, pixel_spacing)."""
    if not _HAS_PYDICOM:
        raise ImportError("pydicom required")
    files = glob.glob(os.path.join(series_dir, "*.dcm"))
    slices = [pydicom.dcmread(f) for f in files]
    slices = [s for s in slices if hasattr(s, "ImagePositionPatient")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    vol = vol * slope + intercept                         # -> HU
    zpos = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    spacing = (float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]))
    return vol, zpos, spacing


def apply_window(vol_hu, level=LUNG_LEVEL, width=LUNG_WIDTH):
    lo, hi = level - width / 2, level + width / 2
    v = np.clip(vol_hu, lo, hi)
    return (v - lo) / (hi - lo)                           # [0,1]


def crop_roi(slice_2d, center_yx, size=ROI):
    h, w = slice_2d.shape
    cy, cx = center_yx
    y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    x0 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    patch = slice_2d[y0:y0 + size, x0:x0 + size]
    if patch.shape != (size, size):                       # pad if near border
        out = np.zeros((size, size), np.float32)
        out[:patch.shape[0], :patch.shape[1]] = patch
        patch = out
    return patch


def gtv_centroid(mask_volume):
    """Centroid (z, y, x) of a binary GTV mask volume [Z,H,W]."""
    zs, ys, xs = np.where(mask_volume > 0)
    if len(zs) == 0:
        raise ValueError("empty GTV mask")
    return int(zs.mean()), (int(ys.mean()), int(xs.mean()))


def make_2p5d(vol_windowed, center_zyx, k=K_SLICES, roi=ROI):
    """Stack k adjacent axial slices around the GTV centroid -> (k, roi, roi)."""
    z, (cy, cx) = center_zyx[0], (center_zyx[1], center_zyx[2])
    half = k // 2
    chans = []
    for dz in range(-half, half + 1):
        zz = int(np.clip(z + dz, 0, vol_windowed.shape[0] - 1))
        chans.append(crop_roi(vol_windowed[zz], (cy, cx), roi))
    return np.stack(chans).astype(np.float32)


def preprocess_patient(series_dir: str, mask_volume=None, center_zyx=None,
                       level=LUNG_LEVEL, width=LUNG_WIDTH, k=K_SLICES, roi=ROI):
    """End-to-end: returns (k, roi, roi) float32 tensor in [0,1].
    Provide either a GTV `mask_volume` (preferred) or an explicit `center_zyx`."""
    vol, zpos, spacing = load_ct_volume(series_dir)
    vw = apply_window(vol, level, width)
    if center_zyx is None:
        if mask_volume is None:
            # fallback: central slice, image centre (documented limitation)
            z = vw.shape[0] // 2
            center_zyx = (z, vw.shape[1] // 2, vw.shape[2] // 2)
        else:
            zc, (yc, xc) = gtv_centroid(mask_volume)
            center_zyx = (zc, yc, xc)
    return make_2p5d(vw, center_zyx, k=k, roi=roi)


# NOTE: parsing RTSTRUCT/SEG GTV masks -> binary volume requires rt-utils or
# pydicom-seg; see data/tcia_download.py for the helper that materialises masks
# aligned to each CT series. Kept separate to isolate the heavy deps.

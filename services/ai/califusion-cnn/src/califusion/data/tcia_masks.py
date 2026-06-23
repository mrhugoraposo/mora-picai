"""
califusion.data.tcia_masks  —  materialize GTV binary masks from TCIA segmentations.

Lung1 (NSCLC-Radiomics) ships a GTV-1 contour in an **RTSTRUCT** per patient
(rt-utils) plus a derived multi-segment **SEG** (pydicom-seg). The canonical primary
gross-tumor volume used by Aerts et al. is RTSTRUCT `GTV-1`; we use it as primary and
keep SEG as a documented fallback.

Orientation guard (important): rt-utils builds the mask in its own internal slice
order, while `dicom_preprocess.load_ct_volume` sorts axial slices by ascending
ImagePositionPatient[z]. These usually agree, but we do NOT assume it. A GTV is, by
construction, solid/soft tissue (HU ≫ aerated lung). We therefore verify alignment by
comparing the mean HU *inside* the candidate mask against its z-flip and keep the
orientation whose interior reads as tissue, not air. This is physically robust and
self-correcting per patient; any applied flip is reported for the QC log.

Leakage note: mask materialization is purely geometric (no label/data fitting).
"""
from __future__ import annotations
import glob
import os
import numpy as np

# soft-tissue HU target a GTV interior should sit near (tumor core ~ +20..+40,
# GTV-1 averages lower due to peritumoral partial-volume; anything ≫ lung is fine)
_TISSUE_HU = 0.0
_LUNG_HU = -700.0


def list_roi_names(ct_dir: str, rtstruct_path: str):
    from rt_utils import RTStructBuilder
    rt = RTStructBuilder.create_from(dicom_series_path=ct_dir, rt_struct_path=rtstruct_path)
    return rt, rt.get_roi_names()


def pick_gtv_roi(names):
    """Prefer exact GTV-1/GTV1/GTV, else first ROI whose name contains 'gtv'."""
    norm = {n: n.strip().lower() for n in names}
    for target in ("gtv-1", "gtv1", "gtv"):
        for n, ln in norm.items():
            if ln == target:
                return n
    cand = [n for n, ln in norm.items() if "gtv" in ln]
    return cand[0] if cand else None


def materialize_gtv_mask(ct_dir: str, rtstruct_path: str, vol_hu: np.ndarray,
                         roi_name: str | None = None):
    """Return (mask[Z,H,W] uint8, info dict). `vol_hu` is the CT volume from
    dicom_preprocess.load_ct_volume (ascending-z), used only for the orientation guard.

    Raises ValueError if no GTV ROI is found or the mask is empty.
    """
    rt, names = list_roi_names(ct_dir, rtstruct_path)
    chosen = roi_name or pick_gtv_roi(names)
    if chosen is None:
        raise ValueError(f"no GTV ROI among {names}")
    m = rt.get_roi_mask_by_name(chosen)               # (H, W, Z) bool
    mask = np.transpose(m, (2, 0, 1)).astype(np.uint8)  # -> (Z, H, W)
    if mask.shape != vol_hu.shape:
        raise ValueError(f"mask {mask.shape} != CT {vol_hu.shape}")
    if mask.sum() == 0:
        raise ValueError(f"empty GTV mask for ROI {chosen!r}")

    def _inside_hu(msk):
        zz, yy, xx = np.where(msk > 0)
        return float(np.mean(vol_hu[zz, yy, xx]))

    hu_asis = _inside_hu(mask)
    hu_flip = _inside_hu(mask[::-1])
    flipped = abs(hu_asis - _TISSUE_HU) > abs(hu_flip - _TISSUE_HU)
    if flipped:
        mask = mask[::-1].copy()
    info = {
        "roi_names": names, "gtv_roi": chosen, "gtv_voxels": int(mask.sum()),
        "hu_inside_asis": round(hu_asis, 1), "hu_inside_flip": round(hu_flip, 1),
        "z_flipped": bool(flipped),
        # sanity flag for QC: interior should read as tissue, far from lung air
        "alignment_ok": bool(abs((_inside_hu(mask)) - _LUNG_HU) > 300.0),
    }
    return mask, info


def seg_foreground_voxels(seg_path: str) -> int:
    """Cross-check helper: total foreground voxels across all SEG segments."""
    import pydicom
    import pydicom_seg
    seg = pydicom_seg.SegmentReader().read(pydicom.dcmread(seg_path))
    return sum(int((seg.segment_data(s) > 0).sum()) for s in seg.available_segments)


def find_series_dirs(patient_dir: str):
    """Locate CT dir and RTSTRUCT file under data/raw/<collection>/<patient>/.
    Layout: <patient>/CT_*/<*.dcm>, <patient>/RTSTRUCT_*/<*.dcm>, <patient>/SEG_*/...
    """
    ct = sorted(glob.glob(os.path.join(patient_dir, "CT_*")))
    rt = sorted(glob.glob(os.path.join(patient_dir, "RTSTRUCT_*", "*.dcm")))
    seg = sorted(glob.glob(os.path.join(patient_dir, "SEG_*", "*.dcm")))
    return (ct[0] if ct else None,
            rt[0] if rt else None,
            seg[0] if seg else None)

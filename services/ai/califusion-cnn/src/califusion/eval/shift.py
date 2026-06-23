"""
califusion.eval.shift
Dataset-shift stress tests applied to a TRAINED model at inference (no retraining),
so degradation is attributable to the shift. Image transforms operate on tensors;
clinical/prevalence transforms operate on arrays and are usable by the tabular arm.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# --------------------------- image degradation ---------------------------- #
def add_gaussian_noise(x, sigma: float):
    """x: tensor in normalised CT space. sigma in same units (e.g. 0.05-0.25)."""
    return x + torch.randn_like(x) * sigma


def gaussian_blur(x, kernel: int = 5, sigma: float = 1.5):
    """Separable Gaussian blur for 2.5D/3D tensors (N,C,H,W)."""
    k = torch.arange(kernel, dtype=x.dtype, device=x.device) - kernel // 2
    g = torch.exp(-(k ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
    kern2d = (g[:, None] * g[None, :])[None, None]
    c = x.shape[1]
    kern2d = kern2d.expand(c, 1, kernel, kernel)
    return F.conv2d(x, kern2d, padding=kernel // 2, groups=c)


def resolution_reduction(x, factor: int = 2):
    """Downsample then upsample to simulate lower acquisition resolution."""
    n, c, h, w = x.shape
    small = F.interpolate(x, scale_factor=1 / factor, mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def jpeg_like_compression(x, q: float = 0.3):
    """Cheap proxy for compression artefacts via block-wise quantisation."""
    levels = max(2, int(2 + q * 30))
    lo, hi = x.amin(), x.amax()
    xn = (x - lo) / (hi - lo + 1e-7)
    xq = torch.round(xn * (levels - 1)) / (levels - 1)
    return xq * (hi - lo) + lo


IMAGE_SHIFTS = {
    "noise_light":  lambda x: add_gaussian_noise(x, 0.05),
    "noise_heavy":  lambda x: add_gaussian_noise(x, 0.20),
    "blur":         lambda x: gaussian_blur(x, 5, 1.5),
    "lowres":       lambda x: resolution_reduction(x, 2),
    "compression":  lambda x: jpeg_like_compression(x, 0.3),
}


# --------------------------- clinical missingness ------------------------- #
def mask_clinical(X: np.ndarray, frac: float, missing_value=np.nan, seed: int = 0, cols=None):
    """Randomly set `frac` of clinical entries to missing_value (pre-imputation).
    If `cols` is given, only mask those column indices (e.g. clinically key vars)."""
    rng = np.random.default_rng(seed)
    Xm = X.astype(float).copy()
    n, d = Xm.shape
    target_cols = np.arange(d) if cols is None else np.asarray(cols)
    mask = rng.random((n, len(target_cols))) < frac
    Xm[:, target_cols] = np.where(mask, missing_value, Xm[:, target_cols])
    return Xm


CLINICAL_MISSING = {"missing_10": 0.10, "missing_30": 0.30, "missing_50": 0.50}


# ----------------------------- prevalence shift --------------------------- #
def resample_to_prevalence(y, p, target_prev: float, seed: int = 0):
    """Resample (with replacement within class) so positive prevalence == target.
    Returns (y_shift, p_shift) for re-evaluation under label shift."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    n = len(y)
    n_pos = int(round(target_prev * n)); n_neg = n - n_pos
    s_pos = rng.choice(pos_idx, n_pos, replace=True)
    s_neg = rng.choice(neg_idx, n_neg, replace=True)
    idx = np.concatenate([s_pos, s_neg]); rng.shuffle(idx)
    return y[idx], p[idx]


PREVALENCE_TARGETS = {"prev_0.30": 0.30, "prev_0.50": 0.50, "prev_0.70": 0.70}

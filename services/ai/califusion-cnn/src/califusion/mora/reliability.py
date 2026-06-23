"""
califusion.mora.reliability — label-free per-modality typicality → reliability.

Class-conditional Mahalanobis typicality (Lee 2018) with a shrunk shared covariance
(Ledoit-Wolf — robust when d is large relative to n). Fit on SOURCE TRAIN only; applied
to any target at test time with no labels. Reliability r(x) ∈ (0,1] is a monotone-
decreasing map of atypicality, calibrated so the bulk of source maps near 1.

Used per modality (r_img, r_clin) AND on the concatenated representation (r_global) so
H2 can compare per-modality vs a single global signal under shift.
"""
from __future__ import annotations
import numpy as np
from sklearn.covariance import LedoitWolf


class MahalanobisReliability:
    def __init__(self, scale_q: float = 0.5):
        # atypicality scale = source-train quantile of s at which r ~ exp(-0.5)
        self.scale_q = scale_q

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = {c: X[y == c].mean(0) for c in self.classes_}
        # pooled within-class covariance (shrunk)
        Xc = np.vstack([X[y == c] - self.means_[c] for c in self.classes_])
        self.cov_ = LedoitWolf().fit(Xc)
        self.prec_ = self.cov_.precision_
        s = self._maha(X)
        self.s_scale_ = max(np.quantile(s, self.scale_q), 1e-6)
        self.s_ref_ = s  # source-train distribution of s (for percentile reliability)
        return self

    def _maha(self, X):
        X = np.asarray(X, float)
        dists = []
        for c in self.classes_:
            d = X - self.means_[c]
            dists.append(np.einsum("ij,jk,ik->i", d, self.prec_, d))
        return np.min(np.vstack(dists), axis=0)  # nearest-class Mahalanobis^2

    def atypicality(self, X):
        return self._maha(X)

    def reliability(self, X):
        """r(x) = exp(-0.5 * s / s_scale) ∈ (0,1], monotone-decreasing in atypicality."""
        s = self._maha(X)
        return np.exp(-0.5 * s / self.s_scale_)


def combine_reliability(r_list, how: str = "min"):
    """Aggregate per-modality reliabilities into one score for deferral ranking."""
    R = np.vstack(r_list)
    if how == "min":
        return R.min(0)
    if how == "hmean":
        return R.shape[0] / np.sum(1.0 / np.clip(R, 1e-6, None), axis=0)
    return R.mean(0)

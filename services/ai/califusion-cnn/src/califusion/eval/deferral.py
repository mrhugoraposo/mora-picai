"""
califusion.eval.deferral
Confidence-aware deferral and accuracy-coverage analysis.

Confidence for binary risk prediction is defined as distance of the calibrated
positive-class probability from the operating threshold t, normalised to [0,1]:
    c_i = 2 * |p_i - t|   (clipped to 1)   when t≈0.5 this is |2p-1| = max-class margin.
A case is RETAINED (acted upon) when c_i >= tau, otherwise DEFERRED to clinician
review. tau is selected ONLY on validation to hit a target coverage.
"""
from __future__ import annotations
import numpy as np


def confidence_binary(p, t: float = 0.5):
    p = np.asarray(p, dtype=float)
    c = 2.0 * np.abs(p - t)
    return np.clip(c, 0.0, 1.0)


def tau_for_coverage(conf, target_coverage: float):
    """Smallest tau (selected on validation conf) achieving <= target coverage retained."""
    conf = np.sort(np.asarray(conf))[::-1]
    k = max(1, int(round(target_coverage * len(conf))))
    k = min(k, len(conf))
    return float(conf[k - 1])


def deferral_report(y, p, t: float, tau: float):
    """Metrics on the retained set at confidence threshold tau."""
    from .metrics import auroc, threshold_metrics
    y = np.asarray(y); p = np.asarray(p)
    conf = confidence_binary(p, t)
    keep = conf >= tau
    n = len(y); n_keep = int(keep.sum())
    coverage = n_keep / n
    yhat = (p >= t).astype(int)
    err_all = float(np.mean(yhat != y))
    out = {
        "tau": float(tau), "coverage": coverage, "deferral_rate": 1 - coverage,
        "n_retained": n_keep, "error_all": err_all,
    }
    if n_keep >= 2 and len(np.unique(y[keep])) == 2:
        out["retained_auroc"] = auroc(y[keep], p[keep])
        tm = threshold_metrics(y[keep], p[keep], t)
        out["retained_error"] = float(np.mean(yhat[keep] != y[keep]))
        out["retained_sensitivity"] = tm["sensitivity"]
        out["retained_specificity"] = tm["specificity"]
    else:
        out.update(dict(retained_auroc=np.nan, retained_error=np.nan,
                        retained_sensitivity=np.nan, retained_specificity=np.nan))
    deferred = ~keep
    n_err = int(np.sum(yhat != y))
    out["frac_errors_deferred"] = float(np.sum((yhat != y) & deferred) / n_err) if n_err else np.nan
    return out


def accuracy_coverage_curve(y, p, t: float, coverages=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """Build Table 5 rows: select tau per coverage on the SAME set (report-only).
    In training, select tau on validation; pass validation conf to tau_for_coverage."""
    conf = confidence_binary(p, t)
    rows = []
    for cov in coverages:
        tau = 0.0 if cov >= 0.999 else tau_for_coverage(conf, cov)
        rows.append(deferral_report(y, p, t, tau))
    return rows

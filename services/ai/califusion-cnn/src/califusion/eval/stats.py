"""
califusion.eval.stats
Statistical-validation utilities for the manuscript: bootstrap confidence
intervals, DeLong test for paired AUROC differences, McNemar test for paired
classification differences, and Holm-Bonferroni multiple-comparison correction.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


# ------------------------------ bootstrap CIs ----------------------------- #
def bootstrap_ci(y, p, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """
    Percentile bootstrap CI for any metric_fn(y, p) -> float.
    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    n = len(y)
    point = metric_fn(y, p)
    stats_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # guard against single-class resamples for AUROC-type metrics
        if len(np.unique(y[idx])) < 2:
            stats_boot[b] = np.nan
            continue
        stats_boot[b] = metric_fn(y[idx], p[idx])
    lo, hi = np.nanpercentile(stats_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def fmt_ci(point, lo, hi, d=3):
    return f"{point:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


# ------------------------------- DeLong test ------------------------------ #
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_roc_test(y, p1, p2):
    """
    DeLong test for two correlated ROC AUCs on the same samples.
    Returns (auc1, auc2, p_value) for H0: AUC1 == AUC2 (two-sided).
    Implementation follows Sun & Xu (2014) fast DeLong.
    """
    y = np.asarray(y); order = (-y).argsort(kind="mergesort")
    label_1_count = int(y.sum())
    preds = np.vstack((p1, p2))[:, order]
    m, n = label_1_count, preds.shape[1] - label_1_count
    pos = preds[:, :m]; neg = preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([_compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([_compute_midrank(preds[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    delongcov = sx / m + sy / n
    l = np.array([[1.0, -1.0]])
    var = float(np.ravel(l @ delongcov @ l.T)[0])
    auc1 = float(np.ravel(aucs)[0]); auc2 = float(np.ravel(aucs)[1])
    if var <= 0:
        return auc1, auc2, 1.0
    z = (auc1 - auc2) / np.sqrt(var)
    pval = 2.0 * stats.norm.sf(abs(z))
    return auc1, auc2, float(pval)


# ------------------------------- McNemar ---------------------------------- #
def mcnemar_test(y, p1, p2, t1: float, t2: float):
    """
    McNemar test on paired correctness at operating thresholds t1, t2.
    Returns (b, c, p_value) where b/c are discordant counts. Uses exact binomial
    for small discordant totals, chi-square (with continuity correction) otherwise.
    """
    y = np.asarray(y)
    c1 = ((np.asarray(p1) >= t1).astype(int) == y)
    c2 = ((np.asarray(p2) >= t2).astype(int) == y)
    b = int(np.sum(c1 & ~c2))   # model1 right, model2 wrong
    c = int(np.sum(~c1 & c2))   # model1 wrong, model2 right
    nd = b + c
    if nd == 0:
        return b, c, 1.0
    if nd < 25:
        p = float(stats.binomtest(min(b, c), nd, 0.5).pvalue)
    else:
        chi2 = (abs(b - c) - 1) ** 2 / nd
        p = float(stats.chi2.sf(chi2, 1))
    return b, c, p


# -------------------------- multiple comparisons -------------------------- #
def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        prev = max(prev, min(val, 1.0))
        adj[idx] = prev
    return adj

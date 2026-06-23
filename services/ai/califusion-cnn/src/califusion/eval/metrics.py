"""
califusion.eval.metrics
Model-agnostic evaluation metrics shared by the tabular, imaging, and fusion arms.

All functions operate on 1-D numpy arrays:
    y    : binary ground-truth labels {0,1}
    p    : predicted probability of the POSITIVE class (high 2-yr mortality risk), in [0,1]

Threshold-independent metrics (AUROC, AUPRC, Brier, NLL, ECE, MCE, calibration
slope/intercept) require only (y, p). Threshold-dependent metrics additionally
take an operating threshold `t` that MUST be selected on validation/calibration
data, never on the evaluation set.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    f1_score, cohen_kappa_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

_EPS = 1e-7


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------- discrimination ----------------------------- #
def auroc(y, p):
    return float(roc_auc_score(y, p))


def auprc(y, p):
    return float(average_precision_score(y, p))


def threshold_metrics(y, p, t: float) -> dict:
    """Sensitivity, specificity, PPV, NPV, F1, kappa at a fixed threshold t."""
    yhat = (np.asarray(p) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "kappa": float(cohen_kappa_score(y, yhat)),
        "threshold": float(t),
    }


def youden_threshold(y, p) -> float:
    """Operating threshold maximising Youden's J (sens + spec - 1). Fit on val only."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


# ------------------------------ calibration ------------------------------- #
def brier(y, p):
    return float(brier_score_loss(y, _clip(p)))


def nll(y, p):
    return float(log_loss(y, _clip(p), labels=[0, 1]))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE over equal-width probability bins (binary, positive-class prob)."""
    p = _clip(p)
    y = np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(y, p, n_bins: int = 10) -> float:
    p = _clip(p); y = np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        mce = max(mce, abs(y[m].mean() - p[m].mean()))
    return float(mce)


def calibration_slope_intercept(y, p):
    """
    Logistic recalibration: logit(y) ~ b0 + b1 * logit(p).
    Perfect calibration => slope b1 = 1, intercept b0 = 0.
    """
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def reliability_bins(y, p, n_bins: int = 10):
    """Return (bin_mean_pred, bin_frac_pos, bin_count) for reliability diagrams."""
    p = _clip(p); y = np.asarray(y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if np.any(m):
            mean_pred.append(p[m].mean()); frac_pos.append(y[m].mean()); counts.append(int(m.sum()))
        else:
            mean_pred.append(np.nan); frac_pos.append(np.nan); counts.append(0)
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


def full_metric_suite(y, p, t: float, n_bins: int = 10) -> dict:
    """Everything needed for Tables 2-4 from a single (y, p, threshold)."""
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "auroc": auroc(y, p), "auprc": auprc(y, p),
        "brier": brier(y, p), "nll": nll(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "mce": maximum_calibration_error(y, p, n_bins),
        "cal_slope": slope, "cal_intercept": intercept,
    }
    out.update(threshold_metrics(y, p, t))
    return out

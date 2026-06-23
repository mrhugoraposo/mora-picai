"""
califusion.calibration.posthoc
Post-hoc probability calibrators fit on a held-out calibration split and applied
at test time. All calibrators take POSITIVE-class probabilities (or logits) as
input so the same code serves the tabular, imaging, and fusion arms.

Calibration never changes the rank order for temperature/Platt (monotone), so
AUROC is preserved up to ties; isotonic is monotone non-decreasing and may merge
ties. This matches the manuscript's claim that calibration improves probability
reliability, not discrimination.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from ..eval.metrics import logit, sigmoid, _clip


class TemperatureScaler:
    """Single-parameter temperature scaling on logits: p' = sigmoid(logit(p)/T)."""
    def __init__(self):
        self.T = 1.0

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "TemperatureScaler":
        z = logit(p_cal)
        y = np.asarray(y_cal, dtype=float)

        def nll(T):
            T = max(T, 1e-3)
            q = _clip(sigmoid(z / T))
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.T = float(res.x)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return sigmoid(logit(p) / self.T)


class PlattScaler:
    """Platt scaling: logistic regression on the logit of the score."""
    def __init__(self):
        self.lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)

    def fit(self, p_cal, y_cal):
        self.lr.fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
        return self

    def transform(self, p):
        return self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class IsotonicScaler:
    """Isotonic regression mapping raw probability -> calibrated probability."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p_cal, y_cal):
        self.iso.fit(_clip(p_cal), np.asarray(y_cal, dtype=float))
        return self

    def transform(self, p):
        return self.iso.predict(_clip(p))


CALIBRATORS = {
    "uncalibrated": None,
    "temperature": TemperatureScaler,
    "platt": PlattScaler,
    "isotonic": IsotonicScaler,
}


def apply_calibrator(name, p_cal, y_cal, p_eval):
    """Fit `name` on (p_cal, y_cal) and return calibrated p_eval. 'uncalibrated' is identity."""
    if name == "uncalibrated":
        return np.asarray(p_eval, dtype=float)
    cal = CALIBRATORS[name]().fit(p_cal, y_cal)
    return cal.transform(p_eval)

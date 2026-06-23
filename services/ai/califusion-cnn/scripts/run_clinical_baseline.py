#!/usr/bin/env python3
"""
scripts/run_clinical_baseline.py
REAL, reproducible clinical-only (tabular) baseline for 2-year OS on Lung1.

Produces the verified CLINICAL-ONLY row of Table 2, the full calibration
comparison (Table 3) for the clinical model, a tabular model-family comparison,
and a reliability diagram. The imaging and fusion rows are produced by the CNN
pipeline (train_image.py / train_multimodal.py) and are intentionally NOT
fabricated here.

Methodology:
  * Patient-level data (one CT per patient) -> stratified evaluation, no leakage.
  * Repeated stratified K-fold (REPEATS x FOLDS) -> out-of-fold (OOF) predictions
    for all usable patients. Within each fold the training folds are further split
    into model-fit and calibration slices; calibrators are fit on the calibration
    slice and applied to the held-out fold. The operating threshold is Youden-J on
    the calibration slice (never the test fold).
  * Metrics computed on pooled OOF predictions per repeat -> mean +/- SD across
    repeats; bootstrap 95% CIs on one representative repeat.

Run: python scripts/run_clinical_baseline.py
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from califusion.data.clinical_preprocess import get_xy, build_preprocessor
from califusion.calibration.posthoc import apply_calibrator, CALIBRATORS
from califusion.eval.metrics import full_metric_suite, auroc, youden_threshold
from califusion.eval.stats import bootstrap_ci, fmt_ci, delong_roc_test, mcnemar_test
from califusion.eval import metrics as M

REPEATS = 5
FOLDS = 5
CALIB_FRAC = 0.20          # of the training folds, held out to fit calibrators
N_BOOT = 1000
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "clinical_baseline")
os.makedirs(OUT, exist_ok=True)


def make_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
        "hist_gboost": HistGradientBoostingClassifier(random_state=0),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                             early_stopping=True, random_state=0),
    }


def oof_predictions(estimator_factory, X, y, repeat_seed):
    """Return dict: calib_method -> OOF prob array, plus OOF threshold array & y order."""
    n = len(y)
    oof = {k: np.full(n, np.nan) for k in CALIBRATORS}
    oof_thr = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=repeat_seed)
    for tr, te in skf.split(X, y):
        fit_idx, cal_idx = train_test_split(
            tr, test_size=CALIB_FRAC, stratify=y[tr], random_state=repeat_seed)
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator_factory())])
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        p_cal = pipe.predict_proba(X.iloc[cal_idx])[:, 1]
        p_te = pipe.predict_proba(X.iloc[te])[:, 1]
        thr = youden_threshold(y[cal_idx], p_cal)
        oof_thr[te] = thr
        for name in CALIBRATORS:
            oof[name][te] = apply_calibrator(name, p_cal, y[cal_idx], p_te)
    return oof, oof_thr


def evaluate_model(factory, X, y, label):
    """Repeated-CV metrics for one model. Returns per-repeat metric dicts + last OOF."""
    per_repeat = {name: [] for name in CALIBRATORS}
    last_oof = None
    for r in range(REPEATS):
        oof, thr = oof_predictions(factory, X, y, repeat_seed=r)
        # one operating threshold per repeat = median of fold thresholds (val-derived)
        t = float(np.nanmedian(thr))
        for name in CALIBRATORS:
            p = oof[name]
            per_repeat[name].append(full_metric_suite(y, p, t))
        last_oof = (oof, t)
    return per_repeat, last_oof


def agg(per_repeat_list, key):
    vals = np.array([d[key] for d in per_repeat_list], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def main():
    print(">> Downloading + labelling Lung1 clinical data ...")
    usable, X, y, ids = get_xy()
    n = len(y); n_pos = int(y.sum())
    print(f"   usable patients = {n} | positive(high-risk) = {n_pos} "
          f"| negative = {n - n_pos} | prevalence = {n_pos/n:.3f}")
    assert n == len(np.unique(ids)), "patient-level uniqueness violated"

    models = make_models()
    family_rows = []
    model_eval = {}
    for mname, mdl in models.items():
        print(f">> Evaluating tabular model: {mname} ({REPEATS}x{FOLDS} CV) ...")
        factory = (lambda m=mdl: __import__("sklearn.base", fromlist=["clone"]).clone(m))
        per_repeat, last_oof = evaluate_model(factory, X, y, mname)
        model_eval[mname] = (per_repeat, last_oof)
        m_auc, s_auc = agg(per_repeat["uncalibrated"], "auroc")
        m_ece_u, _ = agg(per_repeat["uncalibrated"], "ece")
        m_ece_t, _ = agg(per_repeat["temperature"], "ece")
        family_rows.append({
            "model": mname,
            "auroc_mean": round(m_auc, 4), "auroc_sd": round(s_auc, 4),
            "ece_uncal": round(m_ece_u, 4), "ece_temp": round(m_ece_t, 4),
        })

    fam = pd.DataFrame(family_rows).sort_values("auroc_mean", ascending=False)
    fam.to_csv(os.path.join(OUT, "tabular_model_comparison.csv"), index=False)
    print("\n=== Tabular model family (uncalibrated OOF AUROC, mean+/-SD) ===")
    print(fam.to_string(index=False))

    best = fam.iloc[0]["model"]
    print(f"\n>> Selected clinical-only baseline: {best}")
    per_repeat, (oof, t) = model_eval[best]

    # ---- Table 2 clinical-only row (uncalibrated discrimination + calibrated reliability) ----
    metric_keys = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv",
                   "f1", "kappa", "ece", "brier", "nll"]
    row = {"model": f"Clinical-only ({best})"}
    for k in metric_keys:
        mu, sd = agg(per_repeat["uncalibrated"], k)
        row[k] = f"{mu:.3f}±{sd:.3f}"
    # bootstrap CI for AUROC on representative repeat OOF (uncalibrated)
    p_unc = oof["uncalibrated"]
    pt, lo, hi = bootstrap_ci(y, p_unc, auroc, n_boot=N_BOOT, seed=0)
    row["auroc_boot95ci"] = fmt_ci(pt, lo, hi)
    pd.DataFrame([row]).to_csv(os.path.join(OUT, "table2_clinical_row.csv"), index=False)

    # ---- Table 3 calibration comparison (clinical model) ----
    cal_rows = []
    for name in CALIBRATORS:
        r = {"method": name}
        for k in ["ece", "mce", "brier", "nll", "cal_slope", "cal_intercept", "auroc"]:
            mu, sd = agg(per_repeat[name], k)
            r[k] = f"{mu:.3f}±{sd:.3f}"
        cal_rows.append(r)
    pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "table3_calibration.csv"), index=False)

    # ---- DeLong (uncal vs temperature, should be ~ns) + McNemar at threshold ----
    a1, a2, dpval = delong_roc_test(y, oof["uncalibrated"], oof["temperature"])
    b, c, mpval = mcnemar_test(y, oof["uncalibrated"], oof["temperature"], t, t)

    summary = {
        "dataset": "NSCLC-Radiomics (Lung1)",
        "task": "2-year overall survival (binary high-risk)",
        "n_usable": n, "n_positive": n_pos, "n_negative": n - n_pos,
        "prevalence": round(n_pos / n, 4),
        "operating_threshold_median": round(t, 4),
        "selected_model": best,
        "auroc_uncal_boot95ci": fmt_ci(pt, lo, hi),
        "delong_uncal_vs_temperature_p": round(dpval, 4),
        "mcnemar_uncal_vs_temperature_p": round(mpval, 4),
        "cv": f"{REPEATS}x{FOLDS} repeated stratified, nested calibration",
        "n_bootstrap": N_BOOT,
    }
    with open(os.path.join(OUT, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Table 2 (clinical-only row) ===")
    print(pd.DataFrame([row]).to_string(index=False))
    print("\n=== Table 3 (calibration comparison, clinical model) ===")
    print(pd.DataFrame(cal_rows).to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # ---- Reliability diagram (before/after temperature) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mp_u, fp_u, cnt = M.reliability_bins(y, oof["uncalibrated"], 10)
        mp_t, fp_t, _ = M.reliability_bins(y, oof["temperature"], 10)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Perfect")
        ax.plot(mp_u, fp_u, "o-", color="#C05621", label="Uncalibrated")
        ax.plot(mp_t, fp_t, "s-", color="#2E5C8A", label="Temperature")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability — clinical-only ({best}), Lung1 2-yr OS")
        ax.legend(loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "reliability_clinical.png"), dpi=150)
        fig.savefig(os.path.join(OUT, "reliability_clinical.pdf"))
        print("\nWrote reliability_clinical.png/.pdf")
    except Exception as e:
        print("plot skipped:", repr(e)[:120])

    print("\nAll clinical-baseline artifacts written to results/clinical_baseline/")


if __name__ == "__main__":
    main()

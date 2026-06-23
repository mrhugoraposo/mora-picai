"""
califusion.reporting.tables  —  assemble manuscript Tables 2-6.

Merges the VERIFIED clinical-only row (results/clinical_baseline/) with the
imaging/fusion rows produced by train_multimodal.py. Rows not yet computed are
emitted as explicit "[pending GPU run]" placeholders so the manuscript never
shows fabricated numbers.
"""
from __future__ import annotations
import os, glob
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
PENDING = "[pending GPU run]"

TABLE2_MODELS = ["Image-only CNN", "Clinical-only (tabular)", "Early fusion",
                 "Late fusion", "Attention fusion", "CaliFusion-CNN (uncalibrated)",
                 "CaliFusion-CNN (calibrated)"]
TABLE2_COLS = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv",
               "f1", "kappa", "ece", "brier", "nll"]


def load_clinical_row():
    p = os.path.join(RESULTS, "clinical_baseline", "table2_clinical_row.csv")
    return pd.read_csv(p).iloc[0].to_dict() if os.path.exists(p) else None


def build_table2():
    clin = load_clinical_row()
    rows = []
    for name in TABLE2_MODELS:
        row = {"Model": name}
        if name.startswith("Clinical-only") and clin:
            for c in TABLE2_COLS:
                row[c] = clin.get(c, PENDING)
            row["auroc_95ci"] = clin.get("auroc_boot95ci", PENDING)
            row["status"] = "VERIFIED (5x5 CV, n=420)"
        else:
            for c in TABLE2_COLS:
                row[c] = PENDING
            row["auroc_95ci"] = PENDING
            row["status"] = PENDING
        rows.append(row)
    return pd.DataFrame(rows)


def build_table3():
    p = os.path.join(RESULTS, "clinical_baseline", "table3_calibration.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame({"method": ["uncalibrated", "temperature", "platt", "isotonic"]})


def export(df, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    with open(os.path.join(outdir, f"{name}.tex"), "w") as f:
        f.write(df.to_latex(index=False, escape=True))


if __name__ == "__main__":
    out = os.path.join(RESULTS, "manuscript_tables")
    export(build_table2(), "table2_baselines", out)
    export(build_table3(), "table3_calibration", out)
    print(f"Wrote manuscript tables -> {out} (clinical row VERIFIED; others pending GPU run)")

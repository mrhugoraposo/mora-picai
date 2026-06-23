#!/usr/bin/env python3
"""
export_table_s1.py — Supplementary Table S1: per-seed test-split and modality-corruption
counts for the PI-CAI MoRA-under-modality-failure experiment.

Replays the EXACT split and corruption mask used by scripts/picai_sota_baselines.py (the
same data load, the same train_test_split(test_size=0.4, stratify=y, random_state=seed),
and the same per-seed RandomState corruption rng.rand(len(te)) < 0.5) WITHOUT fitting any
model — so every count below is the real value used in the paper, not a re-derivation.

Run:  python scripts/export_table_s1.py
Outputs: results/picai_sota/supplementary_table_s1.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from califusion.data.picai_clinical import load_marksheet, NUMERIC as CLIN

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "picai_sota")
SEEDS = 10

# ---- identical data load to picai_sota_baselines.py ----
rad = pd.read_csv(os.path.join(PROC, "picai_radiomics_features.csv"))
mk = load_marksheet()
mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
df = rad.merge(mk[["patient_id", "label"] + CLIN], on="patient_id", how="inner").reset_index(drop=True)
y = df["label"].to_numpy()
pid = df["patient_id"].astype(str).to_numpy()

rows = []
for seed in range(SEEDS):
    rng = np.random.RandomState(seed)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed)
    yte = y[te]
    broken = rng.rand(len(te)) < 0.5  # EXACT corruption mask used by the experiment
    rows.append({
        "Seed": seed,
        "Test patients": int(len(np.unique(pid[te]))),
        "Test examinations": int(len(te)),
        "Positive": int(yte.sum()),
        "Negative": int((yte == 0).sum()),
        "Corrupted": int(broken.sum()),
        "Uncorrupted": int((~broken).sum()),
        "Corrupted positive": int((broken & (yte == 1)).sum()),
        "Corrupted negative": int((broken & (yte == 0)).sum()),
    })

t = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
t.to_csv(os.path.join(OUT, "supplementary_table_s1.csv"), index=False)
print(f"cohort: {len(df)} examinations · {len(np.unique(pid))} unique patients · "
      f"{int(y.sum())} csPCa-positive ({y.mean():.1%}) · 60/40 stratified split, "
      f"imaging corrupted on a random 50% of test examinations")
print(t.to_string(index=False))
print(f"\nwrote {os.path.join(OUT, 'supplementary_table_s1.csv')}")

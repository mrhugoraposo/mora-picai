#!/usr/bin/env python3
"""
scripts/compare_foundation.py  —  Option-3 decision experiment: does a stronger frozen
CT foundation encoder recover an imaging / fusion accuracy contribution for Lung1 2-yr OS,
or do we commit to a reliability/calibration paper?

All models are evaluated on the SAME repeated stratified patient-level CV folds (5x5),
fixed seeds, leakage-safe (scaler + head fit per training fold only; embeddings are frozen
pre-extracted features). Established baselines are reproduced here with the same harness so
every AUROC is directly comparable.

Models
  clinical            : unified 4-feature clinical (age, gender, stage, histology), best of {logreg, hgb}
  radiomics           : 50 GTV radiomics, best of {logreg, hgb}
  cnn25d              : known deep 2.5D ImageNet CNN image-only (~0.42) -- carried as a fixed reference
  foundation_<enc>    : frozen 512-d embedding -> best of {logreg-L2, hgb}  (+ optional MLP)
  fusion_<enc>_concat : early concat [clinical_raw + embedding] -> single model
  fusion_<enc>_late   : late mean of calibrated clinical & foundation-image probs
  fusion_<enc>_stack  : logistic meta-model over OOF {clinical_p, foundation_p}
  fusion_radiomics_late : clinical + radiomics late-mean (established comparator, ~0.609)

Stats: bootstrap 95% CI on pooled OOF preds (seed 0); paired bootstrap dAUROC (fusion vs
clinical) over the SAME patients; DeLong p; per-seed AUROC spread.

Complementarity: corr(foundation_p, clinical_p); error-overlap (does imaging fix cases
clinical gets wrong?); conditional gain of adding embeddings to clinical.

Decision gates (explicit PASS/FAIL):
  G1 foundation image-only > cnn25d (0.42) AND >= radiomics floor (~0.587)
  G2 clinical+foundation fusion beats clinical-only by >= +0.03 AUROC (pref +0.05),
     paired-bootstrap dAUROC 95% CI excludes 0, stable across seeds, calibration not worse
  G3 complementarity evidence (not just re-encoding stage)

Outputs under results/option3/: model_comparison.csv, paired_delta.csv, per_seed.csv,
complementarity.json, gates.json, summary.json.

Run:  ./.venv/bin/python scripts/compare_foundation.py
"""
from __future__ import annotations
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from califusion.eval import metrics as M
from califusion.eval import stats as S

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "option3")

REPEATS, FOLDS = 5, 5
ENCODERS = ["medicalnet", "ctfm", "r3d18"]
CNN25D_AUROC = 0.415          # established negative DL baseline (results/multimodal_*)
RADIOMICS_FLOOR = 0.587       # established radiomics-only anchor
GATE2_MIN, GATE2_PREF = 0.03, 0.05

CNUM, CCAT = ["age"], ["gender", "overall_stage", "histology"]


# ------------------------------- pipelines ------------------------------------ #
def clinical_pre():
    num = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    cat = Pipeline([("i", SimpleImputer(strategy="constant", fill_value="UNK")),
                    ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, CNUM), ("cat", cat, CCAT)])


def clinical_model(kind):
    clf = (LogisticRegression(max_iter=2000, C=0.5) if kind == "logreg"
           else HistGradientBoostingClassifier(random_state=0))
    return Pipeline([("p", clinical_pre()), ("c", clf)])


def array_model(kind, n_feat):
    """Model over a dense numeric matrix (radiomics or embeddings)."""
    if kind == "logreg":
        clf = LogisticRegression(max_iter=5000, C=0.1)   # stronger L2 for high-dim embeddings
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=800,
                            early_stopping=True, random_state=0)
    else:
        clf = HistGradientBoostingClassifier(random_state=0)
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", clf)])


# ----------------------------- OOF prediction --------------------------------- #
def folds_for_seed(y, seed):
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def oof_clinical(Xdf, y, kind, seed):
    oof = np.full(len(y), np.nan)
    for tr, te in folds_for_seed(y, seed):
        m = clone(clinical_model(kind)); m.fit(Xdf.iloc[tr], y[tr])
        oof[te] = m.predict_proba(Xdf.iloc[te])[:, 1]
    return oof


def oof_array(X, y, kind, seed):
    oof = np.full(len(y), np.nan)
    for tr, te in folds_for_seed(y, seed):
        m = clone(array_model(kind, X.shape[1])); m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def oof_stack(p_clin_seed, p_img_seed, y, seed):
    """Logistic meta-model over OOF [clinical_p, img_p], fit per fold on the SAME seed split."""
    Z = np.column_stack([p_clin_seed, p_img_seed])
    oof = np.full(len(y), np.nan)
    for tr, te in folds_for_seed(y, seed):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(Z[tr], y[tr])
        oof[te] = meta.predict_proba(Z[te])[:, 1]
    return oof


def mean_seed_auroc(make_oof, y):
    a = [roc_auc_score(y, make_oof(s)) for s in range(REPEATS)]
    return float(np.mean(a)), float(np.std(a)), a


# ------------------------------- complementarity ------------------------------ #
def error_overlap(y, p_clin, p_img, p_fus, t=0.5):
    yc = (p_clin >= t).astype(int)
    yi = (p_img >= t).astype(int)
    yf = (p_fus >= t).astype(int)
    clin_wrong = yc != y
    img_right = yi == y
    fus_right = yf == y
    return {
        "clinical_acc": float((yc == y).mean()),
        "foundation_acc": float((yi == y).mean()),
        "fusion_acc": float((yf == y).mean()),
        "clin_wrong_n": int(clin_wrong.sum()),
        "img_fixes_clin_wrong": int((clin_wrong & img_right).sum()),
        "fusion_fixes_clin_wrong": int((clin_wrong & fus_right).sum()),
        "frac_clin_errors_fixed_by_fusion": float((clin_wrong & fus_right).sum() / max(1, clin_wrong.sum())),
    }


# ----------------------------------- main ------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    clin = pd.read_csv(os.path.join(PROC, "clinical_unified.csv"))
    rad = pd.read_csv(os.path.join(PROC, "radiomics_features.csv"))

    # available encoders (only those that actually extracted)
    enc_avail = {}
    for e in ENCODERS:
        p = os.path.join(PROC, f"foundation_embeddings_{e}.npz")
        if os.path.exists(p):
            enc_avail[e] = np.load(p)
    if not enc_avail:
        raise SystemExit("no foundation embeddings found; run extract_foundation_embeddings.py first")

    # Common patient set: clinical labels INTERSECT all available embedding sets INTERSECT radiomics.
    pid_sets = [set(clin.PatientID)] + [set(d.files) for d in enc_avail.values()] + [set(rad.PatientID)]
    common = sorted(set.intersection(*pid_sets))
    clin = clin[clin.PatientID.isin(common)].set_index("PatientID").loc[common].reset_index()
    rad = rad[rad.PatientID.isin(common)].set_index("PatientID").loc[common].reset_index()
    y = clin["label"].to_numpy(int)
    Xclin = clin[CNUM + CCAT]
    rad_cols = [c for c in rad.columns if c not in ("PatientID", "label")]
    Xrad = rad[rad_cols].to_numpy(float)
    emb = {e: np.stack([d[pid] for pid in common]).astype(float) for e, d in enc_avail.items()}

    print(f"common N={len(common)} | prevalence={y.mean():.3f} | encoders={list(enc_avail)}")
    print(f"clinical feats=4 | radiomics={len(rad_cols)} | emb dim={ {e:v.shape[1] for e,v in emb.items()} }")

    rows = []          # model_comparison rows
    oof0 = {}          # seed-0 pooled OOF preds for stats/complementarity
    per_seed = []      # per-seed AUROC table

    def record(name, mean, sd, seeds, oof_seed0, kind=""):
        # bootstrap CI + calibration on seed-0 pooled OOF
        pt, lo, hi = S.bootstrap_ci(y, oof_seed0, M.auroc, n_boot=2000, seed=0)
        brier = M.brier(y, oof_seed0)
        ece = M.expected_calibration_error(y, oof_seed0)
        rows.append({"model": name, "kind": kind, "auroc_cvmean": round(mean, 4),
                     "auroc_cvsd": round(sd, 4), "auroc_boot": round(pt, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "brier": round(brier, 4), "ece": round(ece, 4)})
        oof0[name] = oof_seed0
        for s, a in enumerate(seeds):
            per_seed.append({"model": name, "seed": s, "auroc": round(a, 4)})
        print(f"  {name:26s} AUROC {mean:.4f}±{sd:.4f}  boot {pt:.3f} [{lo:.3f},{hi:.3f}]"
              f"  Brier {brier:.3f}  ECE {ece:.3f}")

    # ---- clinical-only (pick best model by CV mean) ----
    print("\n[clinical]")
    best_c = None
    for k in ("logreg", "gboost"):
        m, sd, seeds = mean_seed_auroc(lambda s, k=k: oof_clinical(Xclin, y, k, s), y)
        if best_c is None or m > best_c[1]:
            best_c = (k, m, sd, seeds)
    ck = best_c[0]
    pclin0 = oof_clinical(Xclin, y, ck, 0)
    record(f"clinical[{ck}]", best_c[1], best_c[2], best_c[3], pclin0, "clinical")
    clinical_name = f"clinical[{ck}]"

    # ---- radiomics-only ----
    print("\n[radiomics]")
    best_r = None
    for k in ("logreg", "gboost"):
        m, sd, seeds = mean_seed_auroc(lambda s, k=k: oof_array(Xrad, y, k, s), y)
        if best_r is None or m > best_r[1]:
            best_r = (k, m, sd, seeds)
    rk = best_r[0]
    prad0 = oof_array(Xrad, y, rk, 0)
    record(f"radiomics[{rk}]", best_r[1], best_r[2], best_r[3], prad0, "radiomics")

    # ---- cnn25d fixed reference (no CI; carried as known negative) ----
    rows.append({"model": "cnn25d_imagenet(ref)", "kind": "image_dl",
                 "auroc_cvmean": CNN25D_AUROC, "auroc_cvsd": None, "auroc_boot": CNN25D_AUROC,
                 "ci_lo": None, "ci_hi": None, "brier": None, "ece": None})
    print(f"\n[cnn25d ref]  AUROC {CNN25D_AUROC} (established negative DL baseline)")

    # ---- radiomics+clinical late-mean comparator ----
    print("\n[fusion_radiomics_late]")
    def rad_late(s):
        pc = oof_clinical(Xclin, y, ck, s); pr = oof_array(Xrad, y, rk, s)
        return 0.5 * (pc + pr)
    m, sd, seeds = mean_seed_auroc(rad_late, y)
    record("fusion_radiomics_late", m, sd, seeds, rad_late(0), "fusion_comparator")

    # ---- per-encoder: foundation image-only + 3 fusions ----
    foundation_names, fusion_names = {}, {}
    for e in emb:
        Xe = emb[e]
        print(f"\n[foundation:{e}]")
        best_f = None
        for k in ("logreg", "gboost", "mlp"):
            m, sd, seeds = mean_seed_auroc(lambda s, k=k: oof_array(Xe, y, k, s), y)
            if best_f is None or m > best_f[1]:
                best_f = (k, m, sd, seeds)
        fk = best_f[0]
        pimg0 = oof_array(Xe, y, fk, 0)
        fname = f"foundation_{e}[{fk}]"
        record(fname, best_f[1], best_f[2], best_f[3], pimg0, "foundation_image")
        foundation_names[e] = fname

        # early concat: clinical raw + embedding (single hgb over mixed; clinical via its own pre)
        print(f"[fusion:{e}:concat]")
        def concat_oof(s, Xe=Xe):
            oof = np.full(len(y), np.nan)
            for tr, te in folds_for_seed(y, s):
                # transform clinical per-fold, then hstack embeddings, fit hgb
                pre = clone(clinical_pre()); Xc_tr = pre.fit_transform(Xclin.iloc[tr]); Xc_te = pre.transform(Xclin.iloc[te])
                sc = StandardScaler().fit(Xe[tr])
                Z_tr = np.hstack([Xc_tr, sc.transform(Xe[tr])])
                Z_te = np.hstack([Xc_te, sc.transform(Xe[te])])
                clf = HistGradientBoostingClassifier(random_state=0); clf.fit(Z_tr, y[tr])
                oof[te] = clf.predict_proba(Z_te)[:, 1]
            return oof
        m, sd, seeds = mean_seed_auroc(concat_oof, y)
        record(f"fusion_{e}_concat", m, sd, seeds, concat_oof(0), "fusion")

        # late mean of calibrated probs (clinical + foundation)
        print(f"[fusion:{e}:late]")
        def late_oof(s, Xe=Xe, fk=fk):
            pc = oof_clinical(Xclin, y, ck, s); pi = oof_array(Xe, y, fk, s)
            return 0.5 * (pc + pi)
        m, sd, seeds = mean_seed_auroc(late_oof, y)
        record(f"fusion_{e}_late", m, sd, seeds, late_oof(0), "fusion")

        # stacking meta-model over OOF clinical & foundation
        print(f"[fusion:{e}:stack]")
        def stack_oof(s, Xe=Xe, fk=fk):
            pc = oof_clinical(Xclin, y, ck, s); pi = oof_array(Xe, y, fk, s)
            return oof_stack(pc, pi, y, s)
        m, sd, seeds = mean_seed_auroc(stack_oof, y)
        sname = f"fusion_{e}_stack"
        record(sname, m, sd, seeds, stack_oof(0), "fusion")
        # track the best fusion per encoder for gate/complementarity (by CV mean)
        cands = [f"fusion_{e}_concat", f"fusion_{e}_late", sname]
        best_fus = max(cands, key=lambda nm: next(r["auroc_cvmean"] for r in rows if r["model"] == nm))
        fusion_names[e] = best_fus

    # ----------------------------- paired stats ----------------------------- #
    print("\n=== paired dAUROC (fusion - clinical), seed-0 pooled OOF, paired bootstrap ===")
    clin_auroc_cvmean = next(r["auroc_cvmean"] for r in rows if r["model"] == clinical_name)
    paired = []
    rng = np.random.default_rng(0)
    n = len(y)
    for e, fus in fusion_names.items():
        pf = oof0[fus]
        # paired bootstrap on same indices
        deltas = np.empty(2000)
        for b in range(2000):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                deltas[b] = np.nan; continue
            deltas[b] = roc_auc_score(y[idx], pf[idx]) - roc_auc_score(y[idx], pclin0[idx])
        dlo, dhi = np.nanpercentile(deltas, [2.5, 97.5])
        dpt = roc_auc_score(y, pf) - roc_auc_score(y, pclin0)
        _, _, pval = S.delong_roc_test(y, pf, pclin0)
        # seed spread of the delta (CV-mean fusion vs CV-mean clinical per seed)
        fus_seeds = [d["auroc"] for d in per_seed if d["model"] == fus]
        clin_seeds = [d["auroc"] for d in per_seed if d["model"] == clinical_name]
        seed_deltas = np.array(fus_seeds) - np.array(clin_seeds)
        paired.append({"encoder": e, "fusion_model": fus,
                       "delta_auroc_boot": round(float(dpt), 4),
                       "ci_lo": round(float(dlo), 4), "ci_hi": round(float(dhi), 4),
                       "delong_p": round(float(pval), 4),
                       "delta_cvmean": round(float(seed_deltas.mean()), 4),
                       "delta_seed_sd": round(float(seed_deltas.std()), 4),
                       "delta_seed_min": round(float(seed_deltas.min()), 4),
                       "delta_seed_max": round(float(seed_deltas.max()), 4),
                       "ci_excludes_0": bool(dlo > 0)})
        print(f"  {e:12s} {fus:24s} dAUROC {dpt:+.4f} [{dlo:+.4f},{dhi:+.4f}]"
              f"  DeLong p={pval:.3f}  seedΔ {seed_deltas.mean():+.4f}±{seed_deltas.std():.4f}")

    # --------------------------- complementarity ---------------------------- #
    print("\n=== complementarity (seed-0 OOF) ===")
    comp = {}
    for e in emb:
        pimg = oof0[foundation_names[e]]
        pfus = oof0[fusion_names[e]]
        corr = float(np.corrcoef(pimg, pclin0)[0, 1])
        eo = error_overlap(y, pclin0, pimg, pfus)
        comp[e] = {"corr_img_clinical": round(corr, 3), **eo}
        print(f"  {e:12s} corr(img,clin)={corr:+.3f}  fusion fixes "
              f"{eo['fusion_fixes_clin_wrong']}/{eo['clin_wrong_n']} clinical errors "
              f"({eo['frac_clin_errors_fixed_by_fusion']:.2f})  "
              f"acc clin {eo['clinical_acc']:.3f} -> fus {eo['fusion_acc']:.3f}")

    # ------------------------------- gates ---------------------------------- #
    print("\n=== DECISION GATES ===")
    best_enc_by_fus = max(fusion_names, key=lambda e: next(
        r["auroc_cvmean"] for r in rows if r["model"] == fusion_names[e]))
    best_found_auroc = max(next(r["auroc_cvmean"] for r in rows if r["model"] == foundation_names[e]) for e in emb)
    best_fus_auroc = next(r["auroc_cvmean"] for r in rows if r["model"] == fusion_names[best_enc_by_fus])
    best_paired = next(p for p in paired if p["encoder"] == best_enc_by_fus)

    g1 = bool(best_found_auroc > CNN25D_AUROC and best_found_auroc >= RADIOMICS_FLOOR - 0.005)
    delta = best_fus_auroc - clin_auroc_cvmean
    # calibration not materially worse: fusion ECE <= clinical ECE + 0.03
    clin_ece = next(r["ece"] for r in rows if r["model"] == clinical_name)
    fus_ece = next(r["ece"] for r in rows if r["model"] == fusion_names[best_enc_by_fus])
    cal_ok = bool(fus_ece <= clin_ece + 0.03)
    g2 = bool(delta >= GATE2_MIN and best_paired["ci_excludes_0"]
              and best_paired["delta_seed_min"] > 0 and cal_ok)
    # complementarity: low-moderate corr AND fusion materially fixes clinical errors beyond noise
    best_comp = comp[best_enc_by_fus]
    g3 = bool(best_comp["corr_img_clinical"] < 0.85
              and best_comp["fusion_acc"] > best_comp["clinical_acc"]
              and best_comp["frac_clin_errors_fixed_by_fusion"] > 0.10)

    gates = {
        "best_encoder_by_fusion": best_enc_by_fus,
        "G1_foundation_beats_cnn_and_radiomics": {
            "pass": g1, "best_foundation_auroc": round(best_found_auroc, 4),
            "cnn25d": CNN25D_AUROC, "radiomics_floor": RADIOMICS_FLOOR},
        "G2_fusion_beats_clinical": {
            "pass": g2, "delta_auroc": round(delta, 4),
            "min_required": GATE2_MIN, "preferred": GATE2_PREF,
            "ci_excludes_0": best_paired["ci_excludes_0"],
            "paired_ci": [best_paired["ci_lo"], best_paired["ci_hi"]],
            "seed_delta_min": best_paired["delta_seed_min"],
            "calibration_ok": cal_ok, "clinical_ece": round(clin_ece, 4), "fusion_ece": round(fus_ece, 4)},
        "G3_complementarity": {
            "pass": g3, "corr_img_clinical": best_comp["corr_img_clinical"],
            "frac_clin_errors_fixed": best_comp["frac_clin_errors_fixed_by_fusion"],
            "clinical_acc": best_comp["clinical_acc"], "fusion_acc": best_comp["fusion_acc"]},
        "ALL_GATES_PASS": bool(g1 and g2 and g3),
    }
    for gk, gv in gates.items():
        if isinstance(gv, dict) and "pass" in gv:
            print(f"  {gk}: {'PASS' if gv['pass'] else 'FAIL'}")
    print(f"  ALL_GATES_PASS: {gates['ALL_GATES_PASS']}")

    # ------------------------------- save ----------------------------------- #
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    pd.DataFrame(paired).to_csv(os.path.join(OUT, "paired_delta.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    with open(os.path.join(OUT, "complementarity.json"), "w") as f:
        json.dump(comp, f, indent=2)
    with open(os.path.join(OUT, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    # encoder weights provenance from extraction manifest
    prov = {}
    mpath = os.path.join(PROC, "foundation_embeddings_manifest.json")
    if os.path.exists(mpath):
        prov = json.load(open(mpath)).get("encoders", {})

    summary = {
        "n": len(common), "prevalence": round(float(y.mean()), 4),
        "cv": f"{REPEATS}x{FOLDS} stratified patient-level", "seeds": list(range(REPEATS)),
        "encoders_used": {e: prov.get(e, {}).get("weights", "?") for e in emb},
        "clinical_model": clinical_name,
        "established_refs": {"cnn25d": CNN25D_AUROC, "radiomics_floor": RADIOMICS_FLOOR},
        "model_comparison": rows, "paired_delta": paired,
        "complementarity": comp, "gates": gates,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> results/option3/{{model_comparison.csv,paired_delta.csv,per_seed.csv,"
          f"complementarity.json,gates.json,summary.json}}")


if __name__ == "__main__":
    main()

"""
mora_engine.py — the live MoRA inference engine for the web console.

Loads the persisted deployment models (built by scripts/build_deployment_models.py) and
runs the Modality-Reliability-Adaptation (MoRA) mechanism on a single case:

  1. per-modality prediction      p_img, p_clin  (source-fit logistic models)
  2. per-modality reliability     r_img, r_clin  (label-free domain-discriminator typicality:
                                   train LR source-vs-this-case, r=clip((1-d)/(1-d_ref),0,1)^λ)
  3. reliability-weighted fusion   p_mora = (r_img*p_img + r_clin*p_clin) / (r_img + r_clin)
  4. decision                      predict / down-weight-imaging / ABSTAIN-and-defer

Every step is returned with its inputs/outputs so the UI can render the exact derivation.
Leakage-safe: all scalers/models/reliability references come from SOURCE; no target labels.
"""
from __future__ import annotations
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# pipeline root (for models/) — resolved relative to this file, overridable via env
_HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.environ.get(
    "CALIFUSION_PIPELINE_ROOT",
    os.path.abspath(os.path.join(_HERE, "..", "..", "ai", "califusion-cnn")),
)
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PIPELINE_ROOT, "models"))
# Fall back to the bundled SYNTHETIC demo assets when no real deployment models are present,
# so the console runs out of the box on non-identifiable example data (see build_demo.py).
if not os.path.exists(os.path.join(MODELS_DIR, "manifest.json")):
    _demo_dir = os.path.abspath(os.path.join(_HERE, "..", "demo_assets"))
    if os.path.exists(os.path.join(_demo_dir, "manifest.json")):
        MODELS_DIR = _demo_dir
EPS = 1e-6

# decision thresholds on reliability (label-free OOD gating)
R_ABSTAIN = 0.35   # if BOTH modalities are this unreliable -> defer to clinician
R_DOWNWEIGHT = 0.6  # if one modality is below this AND the other is healthy -> down-weight it


class ModelsNotBuilt(RuntimeError):
    pass


@lru_cache(maxsize=1)
def manifest() -> dict:
    p = os.path.join(MODELS_DIR, "manifest.json")
    if not os.path.exists(p):
        raise ModelsNotBuilt(
            f"models/manifest.json not found in {MODELS_DIR}. "
            f"Run: python scripts/build_deployment_models.py")
    with open(p) as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _load(name: str) -> dict:
    return joblib.load(os.path.join(MODELS_DIR, name))


@lru_cache(maxsize=2)
def replay_table(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(MODELS_DIR, f"{dataset}_replay.csv"))
    idcol = "patient_id" if dataset == "picai" else "PatientID"
    df[idcol] = df[idcol].astype(str)
    return df


def datasets_available() -> dict:
    """Return per-dataset readiness + metrics for the UI (empty-state friendly)."""
    try:
        man = manifest()
    except ModelsNotBuilt:
        return {}
    return man.get("datasets", {})


# --------------------------------------------------------------------------- reliability
def _reliability(x_scaled: np.ndarray, comp: dict) -> tuple[float, float, float]:
    """Label-free reliability of one scaled case vector against the source reference.

    Trains a fresh logistic domain discriminator on [source (dom 0); this case (dom 1)],
    reads d = P(target | this case), and maps r = clip((1-d)/(1-d_ref),0,1)^λ.
    Returns (r, d, d_ref). r≈1 ⇒ in-distribution (trust); r→0 ⇒ OOD (down-weight)."""
    X_src = comp["X_src_scaled"].astype(np.float64)
    d_ref = float(comp["d_ref"])
    lam = float(comp["lambda"])
    x = np.atleast_2d(x_scaled).astype(np.float64)
    X = np.vstack([X_src, x])
    dom = np.r_[np.zeros(len(X_src)), np.ones(len(x))]
    disc = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, C=0.5))]).fit(X, dom)
    d = float(disc.predict_proba(x)[0, 1])
    r = float(np.clip((1 - d) / (1 - d_ref + EPS), 0.0, 1.0) ** lam)
    return r, d, d_ref


# --------------------------------------------------------------------------- feature prep
def _prep_imaging(dataset: str, raw_feats: dict) -> tuple[np.ndarray, list]:
    art = _load(f"{dataset}_imaging.joblib")
    cols = art["feature_cols"]
    row = pd.DataFrame([{c: raw_feats.get(c, np.nan) for c in cols}])
    row = row.apply(pd.to_numeric, errors="coerce")
    x_scaled = art["scaler"].transform(row)
    return x_scaled, cols


def _prep_clinical(dataset: str, raw_clin: dict) -> np.ndarray:
    art = _load(f"{dataset}_clinical.joblib")
    if dataset == "picai":
        cols = art["feature_cols"]
        row = pd.DataFrame([{c: raw_clin.get(c, np.nan) for c in cols}])
        row = row.apply(pd.to_numeric, errors="coerce")
        return art["scaler"].transform(row)
    # lung1: ColumnTransformer with numeric + categorical
    num = art["numeric"]
    cat = art["categorical"]
    row = {c: raw_clin.get(c, np.nan) for c in num + cat}
    for c in num:
        try:
            row[c] = float(row[c])
        except (TypeError, ValueError):
            row[c] = np.nan
    X = art["preprocessor"].transform(pd.DataFrame([row]))
    return X.toarray() if hasattr(X, "toarray") else X


def _predict_imaging(dataset: str, x_scaled: np.ndarray) -> float:
    return float(_load(f"{dataset}_imaging.joblib")["model"].predict_proba(x_scaled)[0, 1])


def _predict_clinical(dataset: str, x_scaled: np.ndarray) -> float:
    return float(_load(f"{dataset}_clinical.joblib")["model"].predict_proba(x_scaled)[0, 1])


# --------------------------------------------------------------------------- decision
POS_LABEL = {"picai": ("csPCa-positive", "csPCa-negative"),
             "lung1": ("high-risk (death < 2 yr)", "lower-risk (survival >= 2 yr)")}


def _decide(p_mora, r_img, r_clin, thr, dataset="picai"):
    """Map reliabilities + fused risk to an action + human rationale."""
    both_low = (r_img < R_ABSTAIN) and (r_clin < R_ABSTAIN)
    img_bad = r_img < R_DOWNWEIGHT and r_clin >= R_DOWNWEIGHT
    clin_bad = r_clin < R_DOWNWEIGHT and r_img >= R_DOWNWEIGHT
    pos, neg = POS_LABEL.get(dataset, POS_LABEL["picai"])
    label = pos if p_mora >= thr else neg
    if both_low:
        return ("ABSTAIN — defer to clinician",
                "abstain",
                f"Both modalities are out-of-distribution (r_img={r_img:.2f}, r_clin={r_clin:.2f} "
                f"< {R_ABSTAIN}). The case is unlike anything in the training distribution, so the "
                f"model defers rather than risk an unreliable prediction.")
    if img_bad:
        return (f"Predict {label} (imaging down-weighted)",
                "downweight_imaging",
                f"Imaging is out-of-distribution (r_img={r_img:.2f} < {R_DOWNWEIGHT}) while clinical "
                f"is reliable (r_clin={r_clin:.2f}). MoRA automatically shifts trust toward the "
                f"clinical modality; the fused risk {p_mora:.3f} is driven mainly by the clinical arm.")
    if clin_bad:
        return (f"Predict {label} (clinical down-weighted)",
                "downweight_clinical",
                f"Clinical features are out-of-distribution (r_clin={r_clin:.2f} < {R_DOWNWEIGHT}) "
                f"while imaging is reliable (r_img={r_img:.2f}). MoRA shifts trust toward imaging.")
    return (f"Predict {label}",
            "predict",
            f"Both modalities are in-distribution (r_img={r_img:.2f}, r_clin={r_clin:.2f}). MoRA "
            f"fuses them by reliability; fused risk {p_mora:.3f} vs operating threshold {thr:.3f}.")


# --------------------------------------------------------------------------- public API
def infer(dataset: str, imaging_feats: dict | None, clinical_feats: dict,
          case_label=None, source="replay", extraction_note=None) -> dict:
    """Run live MoRA on one case. imaging_feats=None -> clinical-only (imaging unavailable)."""
    if dataset not in ("picai", "lung1"):
        raise ValueError(f"unknown dataset {dataset!r}")
    rel = _load(f"{dataset}_reliability.joblib")
    thr = float(rel["operating_threshold"])

    steps = []
    # ---- clinical modality (always present) ----
    xc = _prep_clinical(dataset, clinical_feats)
    p_clin = _predict_clinical(dataset, xc)
    r_clin, d_clin, dref_clin = _reliability(xc, rel["clinical"])
    steps.append({
        "n": 1, "title": "Clinical modality — prediction",
        "detail": f"Logistic model on {len(rel['clinical_feature_cols']) if dataset=='picai' else 'clinical'} "
                  f"source-fit features → p_clin = {p_clin:.3f}.",
        "inputs": {k: clinical_feats.get(k) for k in
                   (rel["clinical_feature_cols"] if dataset == "picai"
                    else ["age", "gender", "overall_stage", "histology"])},
        "output": {"p_clin": round(p_clin, 4)},
    })
    steps.append({
        "n": 2, "title": "Clinical modality — reliability (label-free)",
        "detail": f"Domain discriminator source-vs-case gives d={d_clin:.3f} (in-distribution level "
                  f"d_ref={dref_clin:.3f}). r_clin = clip((1−d)/(1−d_ref),0,1)^λ = {r_clin:.3f}.",
        "output": {"d": round(d_clin, 4), "d_ref": round(dref_clin, 4), "r_clin": round(r_clin, 4)},
    })

    # ---- imaging modality (optional) ----
    if imaging_feats is not None:
        xi, icols = _prep_imaging(dataset, imaging_feats)
        p_img = _predict_imaging(dataset, xi)
        r_img, d_img, dref_img = _reliability(xi, rel["imaging"])
        n_extracted = sum(1 for c in icols if imaging_feats.get(c) is not None
                          and not (isinstance(imaging_feats.get(c), float)
                                   and np.isnan(imaging_feats.get(c))))
        steps.append({
            "n": 3, "title": "Imaging modality — radiomics → prediction",
            "detail": (extraction_note or
                       f"{n_extracted}/{len(icols)} radiomic features → source-fit logistic model "
                       f"→ p_img = {p_img:.3f}."),
            "output": {"p_img": round(p_img, 4), "n_features": n_extracted},
        })
        steps.append({
            "n": 4, "title": "Imaging modality — reliability (is the scan in-distribution?)",
            "detail": f"Domain discriminator source-vs-scan gives d={d_img:.3f} (d_ref={dref_img:.3f}). "
                      f"r_img = clip((1−d)/(1−d_ref),0,1)^λ = {r_img:.3f}. "
                      f"{'Scan looks typical of training scanners.' if r_img >= R_DOWNWEIGHT else 'Scan is atypical / possible scanner shift.'}",
            "output": {"d": round(d_img, 4), "d_ref": round(dref_img, 4), "r_img": round(r_img, 4)},
        })
    else:
        p_img = None
        r_img = 0.0
        steps.append({
            "n": 3, "title": "Imaging modality — unavailable",
            "detail": "No imaging provided for this case; MoRA proceeds clinical-only "
                      "(r_img set to 0 so the fusion ignores imaging).",
            "output": {"p_img": None, "r_img": 0.0},
        })

    # ---- reliability-weighted fusion ----
    if imaging_feats is not None and (r_img + r_clin) > 0:
        p_mora = (r_img * p_img + r_clin * p_clin) / (r_img + r_clin)
        w_img = r_img / (r_img + r_clin)
        w_clin = r_clin / (r_img + r_clin)
        fuse_detail = (f"p_mora = (r_img·p_img + r_clin·p_clin)/(r_img+r_clin) = "
                       f"({r_img:.2f}·{p_img:.3f} + {r_clin:.2f}·{p_clin:.3f})/({r_img:.2f}+{r_clin:.2f}) "
                       f"= {p_mora:.3f}. Attribution: imaging {w_img:.0%}, clinical {w_clin:.0%}.")
    else:
        p_mora = p_clin
        w_img, w_clin = 0.0, 1.0
        fuse_detail = f"Imaging unavailable → p_mora = p_clin = {p_mora:.3f} (clinical 100%)."
    steps.append({
        "n": 5, "title": "MoRA reliability-weighted fusion",
        "detail": fuse_detail,
        "output": {"p_mora": round(p_mora, 4),
                   "attribution_imaging": round(w_img, 4),
                   "attribution_clinical": round(w_clin, 4)},
    })

    # ---- decision ----
    decision, dcode, rationale = _decide(p_mora, r_img if imaging_feats is not None else 1.0,
                                         r_clin, thr, dataset=dataset)
    steps.append({
        "n": 6, "title": "Decision (modality-attributed)",
        "detail": rationale,
        "output": {"decision": decision, "code": dcode,
                   "threshold": round(thr, 4), "p_mora": round(p_mora, 4)},
    })

    return {
        "dataset": dataset,
        "source": source,
        "p_img": None if p_img is None else round(p_img, 4),
        "p_clin": round(p_clin, 4),
        "r_img": None if imaging_feats is None else round(r_img, 4),
        "r_clin": round(r_clin, 4),
        "p_mora": round(p_mora, 4),
        "attribution_imaging": round(w_img, 4),
        "attribution_clinical": round(w_clin, 4),
        "threshold": round(thr, 4),
        "decision": decision,
        "decision_code": dcode,
        "rationale": rationale,
        "case_label": case_label,
        "extraction_note": extraction_note,
        "steps": steps,
    }


def infer_replay(dataset: str, case_id: str, break_imaging=False, severity=4.0) -> dict:
    """Look up a cached case by id and run MoRA. Optionally corrupt imaging to demo OOD gating."""
    df = replay_table(dataset)
    idcol = "patient_id" if dataset == "picai" else "PatientID"
    labelcol = "label"
    sub = df[df[idcol] == str(case_id)]
    if sub.empty:
        raise KeyError(f"case {case_id!r} not found in {dataset} replay table")
    row = sub.iloc[0]

    img_art = _load(f"{dataset}_imaging.joblib")
    icols = img_art["feature_cols"]
    imaging_feats = {c: (float(row[c]) if c in row and pd.notna(row[c]) else np.nan) for c in icols}

    note = None
    if break_imaging:
        # corrupt raw imaging features by per-feature gaussian noise scaled to the SOURCE std
        # (mirrors the scaled-space corruption in picai_sota_baselines.py without inverse_transform)
        rng = np.random.RandomState(int(abs(hash(str(case_id))) % (2**31)))
        scaler = img_art["scaler"].named_steps["s"]
        std = np.sqrt(np.maximum(scaler.var_, EPS))  # per-feature source std
        noise = severity * rng.standard_normal(len(icols)) * std
        imaging_feats = {c: (imaging_feats[c] + float(noise[i]))
                         for i, c in enumerate(icols)}
        note = (f"Imaging deliberately corrupted (severity {severity}) to simulate a scanner "
                f"shift / acquisition failure — watch r_img collapse and MoRA down-weight imaging.")

    if dataset == "picai":
        clin_cols = _load("picai_clinical.joblib")["feature_cols"]
        clinical_feats = {c: (float(row[c]) if pd.notna(row[c]) else np.nan) for c in clin_cols}
    else:
        clinical_feats = {c: (row[c] if c in row and pd.notna(row[c]) else np.nan)
                          for c in ["age", "gender", "overall_stage", "histology"]}

    label = int(row[labelcol]) if labelcol in row and pd.notna(row[labelcol]) else None
    res = infer(dataset, imaging_feats, clinical_feats, case_label=label,
                source="replay" + ("+broken_imaging" if break_imaging else ""),
                extraction_note=note)
    res["case_id"] = str(case_id)
    if dataset == "picai":
        res["center"] = str(row.get("center", ""))
    return res


def list_cases(dataset: str, n=40) -> list:
    """Return a sample of replay case ids (balanced by label) for the picker."""
    df = replay_table(dataset)
    idcol = "patient_id" if dataset == "picai" else "PatientID"
    out = []
    for lab in (1, 0):
        sub = df[df["label"] == lab].head(n // 2)
        for _, r in sub.iterrows():
            item = {"id": str(r[idcol]), "label": int(r["label"])}
            if dataset == "picai":
                item["center"] = str(r.get("center", ""))
            out.append(item)
    return out

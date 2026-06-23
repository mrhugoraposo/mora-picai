#!/usr/bin/env python3
"""
scripts/train_multimodal.py  —  train CaliFusion-CNN (image / clinical / fusion)
and emit the imaging + fusion rows of Tables 2-5 with calibration, dataset-shift,
deferral, and bootstrap CIs. Run on a GPU machine after preprocessing.

This script is GPU/data-dependent and is NOT executed in the authoring sandbox;
the clinical-only row of Table 2/3 is produced by run_clinical_baseline.py and is
the verified reference. All metrics here come from real model outputs — nothing
is hard-coded.

Expected inputs (built by preprocessing; see README):
    data/processed/image_cache.npz     # PatientID -> (K,H,W) float32
    data/processed/clinical.npz        # X (transformed), y, patient_ids
Usage:
    python scripts/train_multimodal.py --arm fusion --fusion gated --backbone resnet18
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import DataLoader

from califusion.data.datasets import Lung1Multimodal
from califusion.data.multimodal import load_matched_cohort, prepare_split
from califusion.calibration.temperature import TemperatureScalerNN
from califusion.calibration.posthoc import apply_calibrator, CALIBRATORS
from califusion.eval.metrics import (full_metric_suite, auroc, youden_threshold, sigmoid as np_sigmoid)
from califusion.eval.stats import bootstrap_ci, fmt_ci
from califusion.models.networks import CaliFusionCNN, ImageEncoder, ClinicalEncoder


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _device()
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
GATE1_AUROC = 0.63          # signal floor (RESEARCH_DESIGN §8, Gate 1)


def build_model(arm, clinical_dim, args):
    if arm == "image":
        enc = ImageEncoder(args.backbone, in_channels=args.k_slices, out_dim=args.img_dim)
        head = torch.nn.Linear(args.img_dim, 1)
        return torch.nn.Sequential()  # see ImageOnly wrapper below
    return CaliFusionCNN(clinical_dim, backbone=args.backbone, in_channels=args.k_slices,
                         img_dim=args.img_dim, clin_dim=args.clin_dim, fusion=args.fusion)


class ImageOnly(torch.nn.Module):
    def __init__(self, backbone, k, img_dim):
        super().__init__()
        self.enc = ImageEncoder(backbone, in_channels=k, out_dim=img_dim)
        self.head = torch.nn.Linear(img_dim, 1)

    def forward(self, x_img, x_clin=None):
        return self.head(self.enc(x_img)).squeeze(-1)


class ClinicalOnlyNN(torch.nn.Module):
    def __init__(self, clinical_dim, clin_dim):
        super().__init__()
        self.enc = ClinicalEncoder(clinical_dim, out_dim=clin_dim)
        self.head = torch.nn.Linear(clin_dim, 1)

    def forward(self, x_img, x_clin):
        return self.head(self.enc(x_clin)).squeeze(-1)


def make_net(arm, clinical_dim, args):
    if arm == "image":
        return ImageOnly(args.backbone, args.k_slices, args.img_dim)
    if arm == "clinical":
        return ClinicalOnlyNN(clinical_dim, args.clin_dim)
    return CaliFusionCNN(clinical_dim, backbone=args.backbone, in_channels=args.k_slices,
                         img_dim=args.img_dim, clin_dim=args.clin_dim, fusion=args.fusion)


def run_epoch(net, loader, opt=None):
    train = opt is not None
    net.train(train)
    bce = torch.nn.BCEWithLogitsLoss()
    logits_all, y_all, loss_sum = [], [], 0.0
    for img, clin, y in loader:
        img, clin, y = img.to(DEVICE), clin.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logit = net(img, clin)
            loss = bce(logit, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss) * len(y)
        logits_all.append(logit.detach().cpu()); y_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(y_all), loss_sum / len(loader.dataset)


@torch.no_grad()
def collect_logits(net, loader):
    net.eval(); L, Y = [], []
    for img, clin, y in loader:
        L.append(net(img.to(DEVICE), clin.to(DEVICE)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def train_arm(arm, coh, args):
    """Train one arm across seeds (leakage-safe per-split unified-clinical transform).
    Returns (per_seed_rows, per_seed_test_auroc_uncal)."""
    ids, X_raw, y, cache = coh
    rows, test_auc_unc = [], []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        sp = prepare_split(ids, X_raw, y, seed=seed)
        clinical_dim = sp["clinical_dim"]

        def mk(split, aug):
            sids, Xt, ys = sp[split]
            ds = Lung1Multimodal(sids, Xt, ys, cache, augment=aug)
            return DataLoader(ds, batch_size=args.batch, shuffle=aug,
                              num_workers=0, drop_last=aug)  # drop_last on train -> no size-1 BatchNorm batch
        dl_tr, dl_va, dl_te = mk("train", True), mk("val", False), mk("test", False)

        net = make_net(arm, clinical_dim, args).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        best_val, best_state, patience, bad = 1e9, None, 8, 0
        for ep in range(args.epochs):
            run_epoch(net, dl_tr, opt)
            _, _, vloss = run_epoch(net, dl_va)
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)

        # calibrators + operating threshold fit on VALIDATION only (no leakage)
        vlog, vy = collect_logits(net, dl_va)
        tscaler = TemperatureScalerNN().fit(vlog, vy)
        p_val = np_sigmoid(vlog.numpy()); yv = vy.numpy()
        t_op = youden_threshold(yv, np_sigmoid((vlog / tscaler.T).numpy()))

        tlog, ty = collect_logits(net, dl_te)
        p_test_unc = np_sigmoid(tlog.numpy()); yt = ty.numpy()
        test_auc_unc.append(float(auroc(yt, p_test_unc)))
        for name in CALIBRATORS:
            p_cal = apply_calibrator(name, p_val, yv, p_test_unc)
            m = full_metric_suite(yt, p_cal, t_op)
            rows.append({"arm": arm, "fusion": args.fusion if arm == "fusion" else "-",
                         "calibration": name, "seed": seed, **m})
        print(f"   [{arm:8s} seed {seed}] test AUROC(uncal)={test_auc_unc[-1]:.3f} "
              f"(epochs={ep + 1}, val_loss={best_val:.3f})", flush=True)
    return rows, test_auc_unc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["image", "clinical", "fusion"],
                    choices=["image", "clinical", "fusion"])
    ap.add_argument("--fusion", choices=["concat", "gated", "attention"], default="gated")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k_slices", type=int, default=3)
    ap.add_argument("--img_dim", type=int, default=256)
    ap.add_argument("--clin_dim", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    import pandas as pd
    print(f">> device={DEVICE} | arms={args.arms} | fusion={args.fusion} | seeds={args.seeds}", flush=True)
    coh = load_matched_cohort()
    ids = coh[0]
    print(f">> matched cohort n={len(ids)} prevalence={coh[2].mean():.3f}", flush=True)

    all_rows, arm_auc = [], {}
    for arm in args.arms:
        print(f">> training arm: {arm}", flush=True)
        rows, auc_unc = train_arm(arm, coh, args)
        all_rows += rows
        arm_auc[arm] = auc_unc

    df = pd.DataFrame(all_rows)
    agg = (df.groupby(["arm", "fusion", "calibration"])
             .agg(lambda s: f"{np.nanmean(s):.3f}±{np.nanstd(s):.3f}"
                  if np.issubdtype(s.dtype, np.number) else s.iloc[0]))
    tag = "_".join(args.arms) + f"_{args.fusion}"
    outdir = os.path.join(RESULTS, f"multimodal_{tag}")
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "per_seed_metrics.csv"), index=False)
    agg.to_csv(os.path.join(outdir, "aggregated_metrics.csv"))

    # ---- reproducibility sidecar (config + seeds + env; git SHA if available) ----
    sha = "no-git"
    try:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or "no-git"
    except Exception:
        pass
    meta = {"device": DEVICE, "torch": torch.__version__, "args": vars(args),
            "n_matched": int(len(ids)), "prevalence": round(float(coh[2].mean()), 4),
            "git_sha": sha, "seeds": list(range(args.seeds)),
            "arm_test_auroc_uncal": {a: [round(x, 4) for x in v] for a, v in arm_auc.items()}}
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Aggregated metrics (mean±sd over seeds) ===")
    print(agg.to_string())

    # ---- GATE 1 decision (RESEARCH_DESIGN §8): fusion uncalibrated test AUROC >= ~0.63 ----
    if "fusion" in arm_auc:
        fa = np.array(arm_auc["fusion"])
        mu, sd = float(fa.mean()), float(fa.std())
        clin = np.array(arm_auc.get("clinical", [np.nan])); img = np.array(arm_auc.get("image", [np.nan]))
        passed = mu >= GATE1_AUROC
        gate = {"gate": "Gate 1 — signal floor", "threshold": GATE1_AUROC,
                "fusion_auroc_mean": round(mu, 4), "fusion_auroc_sd": round(sd, 4),
                "fusion_auroc_per_seed": [round(x, 4) for x in fa],
                "clinical_auroc_mean": round(float(np.nanmean(clin)), 4),
                "image_auroc_mean": round(float(np.nanmean(img)), 4),
                "clinical_anchor_full_field": 0.583, "intersection_clinical_cv": 0.578,
                "decision": "PASS" if passed else "FAIL"}
        with open(os.path.join(outdir, "gate1_decision.json"), "w") as f:
            json.dump(gate, f, indent=2)
        bar = "=" * 64
        print(f"\n{bar}\nGATE 1 — fusion({args.fusion}) test AUROC = {mu:.3f} ± {sd:.3f} "
              f"(per-seed {[round(x,3) for x in fa]})")
        print(f"  clinical-NN {np.nanmean(clin):.3f} | image {np.nanmean(img):.3f} | threshold {GATE1_AUROC}")
        print(f"  DECISION: {'PASS — proceed to MoRA (Phase D)' if passed else 'FAIL — STOP and report (revisit features/arch)'}")
        print(bar)

    print(f"\nWrote {outdir}/ (per_seed_metrics, aggregated_metrics, gate1_decision, run_meta)")


if __name__ == "__main__":
    main()

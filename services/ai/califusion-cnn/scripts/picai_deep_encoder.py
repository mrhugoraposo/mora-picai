#!/usr/bin/env python3
"""
scripts/picai_deep_encoder.py — DEEP bpMRI encoder for PI-CAI csPCa (decision experiment).

Builds a gland-centered multi-sequence (T2W/ADC/HBV) deep imaging arm to test whether a
TRAINED encoder strengthens the MoRA reliability method vs the existing radiomics arm
(AUROC ~0.773). Leakage-safe by construction:

  PREPROCESS (--preprocess):
    For each case: resample ADC+HBV+whole-gland-mask onto the T2W grid (same pattern as
    scripts/picai_radiomics.py), find the gland centroid, take the K central gland slices
    (ranked by per-slice gland area), crop a FIXED physical FOV (FOV_MM) box centered on the
    centroid, resample each sequence to OUT_HW x OUT_HW, and z-score each sequence WITHIN the
    gland (MRI has no absolute scale). Cache -> data/processed/picai_deep_patches.npz as a
    (3, K, H, W) tensor per patient (channels = t2w, adc, hbv).
    Only intensity/geometry — NO labels touched here.

  TRAIN + EXTRACT (--train):
    Compact 3D CNN csPCa classifier (MONAI resnet18, spatial_dims=3, in=3ch), PATIENT-LEVEL
    5-fold stratified CV, early stopping on a held-out slice of each fold's train split.
    For each fold, the model trains ONLY on that fold's training patients; we then take the
    penultimate-layer (512-d) embedding for the fold's TEST patients. Every case therefore
    gets an OUT-OF-FOLD embedding from a model that never saw it (leakage-safe). We also
    record the deep model's OOF csPCa AUROC. Cache -> data/processed/picai_deep_embeddings.npz
    (keyed by patient_id) + the OOF probabilities (key "__oof_proba__"/"__y__"/"__pid__").

Run:
  ./.venv/bin/python scripts/picai_deep_encoder.py --preprocess            # build patches (slow, once)
  ./.venv/bin/python scripts/picai_deep_encoder.py --train                 # 5-fold train + OOF embeddings
  ./.venv/bin/python scripts/picai_deep_encoder.py --preprocess --train     # both
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(ROOT, "data", "raw", "PI-CAI", "images")
MASKS = os.path.join(ROOT, "data", "raw", "PI-CAI", "picai_labels",
                     "anatomical_delineations", "whole_gland", "AI", "Bosma22b")
PROC = os.path.join(ROOT, "data", "processed")
OUTDIR = os.path.join(ROOT, "results", "picai_deep")
SEQS = ["t2w", "adc", "hbv"]

# Gland-centered ROI: K central gland slices, FOV_MM physical box -> OUT_HW x OUT_HW.
K_SLICES = 12
FOV_MM = 80.0
OUT_HW = 96

PATCH_PATH = os.path.join(PROC, "picai_deep_patches.npz")
EMB_PATH = os.path.join(PROC, "picai_deep_embeddings.npz")


# ----------------------------- preprocessing ----------------------------- #
def resample_to(ref, moving, is_mask=False):
    import SimpleITK as sitk
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(moving)


def find_case_files(case_dir):
    out = {}
    for seq in SEQS:
        hits = glob.glob(os.path.join(case_dir, f"*_{seq}.mha"))
        if hits:
            out[seq] = hits[0]
    return out


def _crop_fov_slice(arr2d, cy, cx, half_y, half_x, out_hw):
    """Crop a (2*half)+ box centered on (cy,cx) from a 2D array, zero-padded, resample to out_hw."""
    import SimpleITK as sitk
    H, W = arr2d.shape
    y0, x0 = int(round(cy - half_y)), int(round(cx - half_x))
    y1, x1 = int(round(cy + half_y)), int(round(cx + half_x))
    box = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y1), min(W, x1)
    box[ys - y0:ye - y0, xs - x0:xe - x0] = arr2d[ys:ye, xs:xe]
    # resample the physical box to out_hw x out_hw via SITK (linear)
    im = sitk.GetImageFromArray(box)
    rs = sitk.ResampleImageFilter()
    rs.SetSize([out_hw, out_hw])
    rs.SetOutputSpacing([box.shape[1] / out_hw, box.shape[0] / out_hw])
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetOutputOrigin(im.GetOrigin()); rs.SetOutputDirection(im.GetDirection())
    return sitk.GetArrayFromImage(rs.Execute(im)).astype(np.float32)


def extract_patch(case_dir):
    """Return (patient_id, study_id, patch[3,K,H,W]) or None. z-scored within gland per sequence."""
    import SimpleITK as sitk
    files = find_case_files(case_dir)
    if "t2w" not in files:
        return None
    base = os.path.basename(files["t2w"]).replace("_t2w.mha", "")
    mask_path = os.path.join(MASKS, base + ".nii.gz")
    if not os.path.exists(mask_path):
        return None
    t2w = sitk.ReadImage(files["t2w"], sitk.sitkFloat32)
    mask_img = resample_to(t2w, sitk.ReadImage(mask_path), is_mask=True)
    m = sitk.GetArrayFromImage(mask_img) > 0           # [Z,H,W]
    if m.sum() < 50:
        return None
    sp = t2w.GetSpacing()                               # (x,y,z) mm
    half_y = (FOV_MM / 2.0) / sp[1]                      # half-box in voxels (in-plane)
    half_x = (FOV_MM / 2.0) / sp[0]

    # gland centroid (in-plane) and central K gland slices by area
    zs, ys, xs = np.where(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    per_slice = m.reshape(m.shape[0], -1).sum(1)
    gland_z = np.where(per_slice > 0)[0]
    if len(gland_z) == 0:
        return None
    zc = int(round(zs.mean()))
    order = sorted(gland_z, key=lambda z: abs(z - zc))   # central gland slices first
    chosen = sorted(order[:K_SLICES])
    if len(chosen) < K_SLICES:                           # pad by repeating nearest gland slices
        chosen = (chosen + [chosen[-1]] * K_SLICES)[:K_SLICES]

    chans = []
    for seq in SEQS:
        if seq not in files:                             # missing sequence -> zeros channel
            chans.append(np.zeros((K_SLICES, OUT_HW, OUT_HW), np.float32))
            continue
        img = sitk.ReadImage(files[seq], sitk.sitkFloat32)
        if seq != "t2w":
            img = resample_to(t2w, img)                  # onto T2W grid
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        gland_vals = arr[m]
        mu, sd = float(gland_vals.mean()), float(gland_vals.std()) + 1e-6
        arr_z = (arr - mu) / sd                          # z-score within gland
        stack = np.stack([_crop_fov_slice(arr_z[z], cy, cx, half_y, half_x, OUT_HW)
                          for z in chosen], axis=0)       # [K,H,W]
        chans.append(stack.astype(np.float32))
    patch = np.stack(chans, axis=0)                       # [3,K,H,W]
    return int(base.split("_")[0]), int(base.split("_")[1]), patch


def run_preprocess(limit=0):
    cases = sorted(d for d in glob.glob(os.path.join(IMG, "*")) if os.path.isdir(d))
    if limit:
        cases = cases[:limit]
    print(f"[preprocess] {len(cases)} case dirs | ROI={3}x{K_SLICES}x{OUT_HW}x{OUT_HW} "
          f"FOV={FOV_MM}mm", flush=True)
    patches, pids, sids, t0 = {}, [], [], time.time()
    fails = 0
    for i, c in enumerate(cases, 1):
        try:
            r = extract_patch(c)
            if r is None:
                fails += 1
            else:
                pid, sid, patch = r
                patches[str(pid)] = patch
                pids.append(pid); sids.append(sid)
        except Exception as e:  # noqa: BLE001
            fails += 1
            if fails <= 8:
                print(f"  skip {os.path.basename(c)}: {repr(e)[:70]}")
        if i % 100 == 0:
            print(f"  ...{i}/{len(cases)} ok={len(patches)} fail={fails} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(PROC, exist_ok=True)
    np.savez_compressed(PATCH_PATH, **patches)
    meta = {"n": len(patches), "fails": fails, "K": K_SLICES, "fov_mm": FOV_MM,
            "out_hw": OUT_HW, "channels": SEQS,
            "patient_ids": pids, "study_ids": sids}
    json.dump(meta, open(os.path.join(PROC, "picai_deep_patches_manifest.json"), "w"), indent=2)
    print(f"[preprocess] cached {len(patches)} patches -> {PATCH_PATH} "
          f"({time.time()-t0:.0f}s, fails={fails})")


# ------------------------------- model ------------------------------- #
# Compact 3D CNN (kept light for MPS, per the build constraint). A 3D ResNet18 is ~100x heavier
# and runs ~6x slower per step on MPS without learning faster here; this 0.3M-param net trains in
# seconds/epoch and learns the csPCa signal. Penultimate = EMB_DIM-d global-avg-pooled feature.
EMB_DIM = 128


def build_model():
    import torch.nn as nn

    class SmallCNN3D(nn.Module):
        def __init__(self, cin=3, emb=EMB_DIM):
            super().__init__()

            def blk(i, o, s):
                return nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False),
                                     nn.BatchNorm3d(o), nn.ReLU(inplace=True), nn.MaxPool3d(s))
            self.f = nn.Sequential(
                blk(cin, 16, (1, 2, 2)),    # [K,96,96] -> [K,48,48]
                blk(16, 32, (2, 2, 2)),     # -> [K/2,24,24]
                blk(32, 64, (2, 2, 2)),     # -> [.,12,12]
                blk(64, emb, (1, 2, 2)),    # -> [.,6,6]
            )
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(emb, 1)

        def feat(self, x):
            return self.pool(self.f(x)).flatten(1)        # (B, EMB_DIM)

        def forward(self, x):
            return self.fc(self.feat(x))                  # (B, 1)

    return SmallCNN3D()


def _penultimate(net, x):
    """EMB_DIM-d global-avg-pooled penultimate feature (pre-fc)."""
    return net.feat(x)


def run_train(seed=0, epochs=40, batch=32, lr=2e-3, patience=8):
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from califusion.data.picai_clinical import load_marksheet

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if not os.path.exists(PATCH_PATH):
        raise SystemExit(f"missing {PATCH_PATH}; run --preprocess first")
    z = np.load(PATCH_PATH)
    pids = sorted(int(k) for k in z.files)
    X = np.stack([z[str(p)] for p in pids]).astype(np.float32)   # [N,3,K,H,W]

    mk = load_marksheet()
    mk["label"] = (mk["case_csPCa"].astype(str).str.upper() == "YES").astype(int)
    lab = mk.set_index("patient_id")["label"].to_dict()
    cen = mk.set_index("patient_id")["center"].astype(str).to_dict()
    keep = [i for i, p in enumerate(pids) if p in lab]
    pids = [pids[i] for i in keep]; X = X[keep]
    y = np.array([lab[p] for p in pids], dtype=np.float32)
    centers = np.array([cen[p] for p in pids])
    print(f"[train] device={dev} N={len(pids)} csPCa+={int(y.sum())} "
          f"({y.mean():.3f}) patch={X.shape[1:]} seed={seed}", flush=True)

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    oof_proba = np.full(len(y), np.nan, dtype=np.float64)
    oof_emb = np.zeros((len(y), EMB_DIM), dtype=np.float32)

    def run_batches(net, idx, train=True, opt=None, lossf=None):
        net.train(train)
        probs = np.zeros(len(idx)); embs = np.zeros((len(idx), EMB_DIM), np.float32)
        order = np.random.permutation(len(idx)) if train else np.arange(len(idx))
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for s in range(0, len(idx), batch):
                bb = order[s:s + batch]; gi = [idx[b] for b in bb]
                xb = Xt[gi].to(dev); yb = yt[gi].to(dev)
                feat = _penultimate(net, xb)
                logit = net.fc(feat).squeeze(1)
                yb = yb.float()
                if train:
                    loss = lossf(logit, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                p = torch.sigmoid(logit).detach().cpu().numpy()
                probs[bb] = p; embs[bb] = feat.detach().cpu().numpy()
        return probs, embs

    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    t0 = time.time()
    for fold, (tr_all, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        torch.manual_seed(seed * 100 + fold); np.random.seed(seed * 100 + fold)
        # inner early-stopping split (patient-level, stratified) from this fold's train
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(tr_all, test_size=0.18, stratify=y[tr_all],
                                  random_state=seed * 100 + fold)
        net = build_model().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        pos_w = torch.tensor([float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum()))],
                             dtype=torch.float32, device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best_auc, best_state, bad = -1.0, None, 0
        tf = time.time()
        for ep in range(epochs):
            run_batches(net, list(tr), train=True, opt=opt, lossf=lossf)
            vp, _ = run_batches(net, list(va), train=False)
            try:
                vauc = roc_auc_score(y[va], vp)
            except ValueError:
                vauc = 0.5
            if vauc > best_auc + 1e-4:
                best_auc = vauc; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            print(f"    [f{fold} ep{ep:02d}] val_auc={vauc:.3f} best={best_auc:.3f} "
                  f"bad={bad} ({(time.time()-tf)/(ep+1):.1f}s/ep)", flush=True)
            if bad >= patience:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        tp, temb = run_batches(net, list(te), train=False)
        oof_proba[te] = tp; oof_emb[te] = temb
        print(f"  fold {fold}: train={len(tr)} val={len(va)} test={len(te)} "
              f"best_val_auc={best_auc:.3f} test_auc={roc_auc_score(y[te], tp):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    oof_auc = float(roc_auc_score(y, oof_proba))
    print(f"[train] OOF csPCa AUROC (deep imaging-only, 5-fold) = {oof_auc:.4f}")

    # cache embeddings keyed by patient_id + OOF arrays for downstream reuse
    save = {str(p): oof_emb[i] for i, p in enumerate(pids)}
    save["__oof_proba__"] = oof_proba.astype(np.float32)
    save["__y__"] = y.astype(np.float32)
    save["__pid__"] = np.array(pids)
    save["__center__"] = centers
    np.savez_compressed(EMB_PATH, **save)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"deep_oof_auroc": round(oof_auc, 4), "n": len(pids),
               "pos": int(y.sum()), "emb_dim": EMB_DIM, "seed": seed,
               "K": K_SLICES, "fov_mm": FOV_MM, "out_hw": OUT_HW,
               "model": "compact 3D CNN (~0.3M params, in=3ch t2w/adc/hbv), BCE pos-weighted, early-stop"},
              open(os.path.join(OUTDIR, "deep_encoder_oof.json"), "w"), indent=2)
    print(f"[train] cached embeddings -> {EMB_PATH}; OOF summary -> "
          f"{OUTDIR}/deep_encoder_oof.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    if not (args.preprocess or args.train):
        ap.error("pass --preprocess and/or --train")
    if args.preprocess:
        run_preprocess(limit=args.limit)
    if args.train:
        run_train(seed=args.seed, epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()

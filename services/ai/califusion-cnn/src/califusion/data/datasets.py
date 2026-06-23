"""
califusion.data.splits / datasets  —  patient-level split + multimodal Dataset.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split


def patient_level_split(patient_ids, y, ratios=(0.70, 0.15, 0.15), seed: int = 0):
    """Stratified train/val/test split with NO patient in more than one split.
    Lung1 has one CT per patient, so row-level stratification is patient-level."""
    ids = np.asarray(patient_ids); y = np.asarray(y)
    idx = np.arange(len(ids))
    r_tr, r_val, r_te = ratios
    tr, tmp = train_test_split(idx, test_size=(r_val + r_te), stratify=y, random_state=seed)
    rel = r_te / (r_val + r_te)
    val, te = train_test_split(tmp, test_size=rel, stratify=y[tmp], random_state=seed)
    assert set(tr).isdisjoint(val) and set(tr).isdisjoint(te) and set(val).isdisjoint(te)
    return tr, val, te


try:
    import torch
    from torch.utils.data import Dataset

    class Lung1Multimodal(Dataset):
        """Returns (image[K,H,W] float32, clinical[D] float32, label float32).

        `image_cache` maps PatientID -> preprocessed (K,H,W) array (built by the
        preprocessing step). `clinical_matrix` is the dense, already-transformed
        clinical feature matrix aligned to `patient_ids`.
        """
        def __init__(self, patient_ids, clinical_matrix, labels, image_cache,
                     augment=False):
            self.ids = list(patient_ids)
            self.clin = np.asarray(clinical_matrix, dtype=np.float32)
            self.y = np.asarray(labels, dtype=np.float32)
            self.cache = image_cache
            self.augment = augment

        def __len__(self):
            return len(self.ids)

        def _augment(self, img):
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy()
            img = img + np.random.randn(*img.shape).astype(np.float32) * 0.01
            return img

        def __getitem__(self, i):
            pid = self.ids[i]
            img = self.cache[pid].astype(np.float32)
            if self.augment:
                img = self._augment(img)
            return (torch.from_numpy(img),
                    torch.from_numpy(self.clin[i]),
                    torch.tensor(self.y[i]))
except Exception:
    pass

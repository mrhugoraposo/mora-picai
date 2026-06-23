"""
califusion.calibration.temperature  —  temperature scaling for NN logits.

Fit a single scalar T>0 on validation logits by minimising NLL; apply at test
time as p = sigmoid(logit / T). Preserves argmax/ranking => AUROC unchanged.
For tabular/sklearn probabilities use califusion.calibration.posthoc instead.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    class TemperatureScalerNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_T = nn.Parameter(torch.zeros(1))   # T = exp(log_T) > 0

        @property
        def T(self):
            return self.log_T.exp().item()

        def forward(self, logits):
            return logits / self.log_T.exp()

        def fit(self, val_logits, val_labels, max_iter: int = 200, lr: float = 0.01):
            val_logits = val_logits.detach().float()
            val_labels = val_labels.detach().float()
            opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter)
            bce = nn.BCEWithLogitsLoss()

            def closure():
                opt.zero_grad()
                loss = bce(self.forward(val_logits), val_labels)
                loss.backward()
                return loss
            opt.step(closure)
            return self
except Exception:
    pass

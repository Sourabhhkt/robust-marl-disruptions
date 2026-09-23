"""
Milestone 10 advanced coordinators (WP2): closing the Milestone 9 gaps.

WP2a -- Heterogeneity-aware graph-attention aggregator. The M9 graph-attention
aggregator keyed only on message deviation and magnitude, so it could not exploit
team heterogeneity (M9 showed it did not beat the plain mean as heterogeneity
grew). This version additionally consumes per-neighbor *link-quality* features --
age-of-information, an estimated transmit reliability, and the advertised message
precision -- so it can down-weight a merely weak (but honest) neighbor, not only
an outlier. Link quality is taken to be observable/advertised (radio statistics,
capability advertisement), the natural deployment assumption.

WP2b -- Byzantine-robust mismatch estimator for distributed dispatch lives in
infra_m10.py (it operates on the dispatch rollout, not the consensus aggregator).
"""
from __future__ import annotations

import os
from typing import Any, List, Optional
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:
    class HeteroAttnNet(nn.Module):
        """Attention over neighbors using value-deviation/magnitude AND link-quality
        features [age, reliability, precision_norm]. Five input features per
        neighbor; scalar attention weight applied to the (vector) message."""
        def __init__(self, hidden: int = 32):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(5, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1))
            self.own_gate = nn.Parameter(torch.tensor(-1.0))

        def forward(self, own, vals, quality):
            # own: (d,)  vals: (k,d)  quality: (k,3) = [age, reliability, precision]
            dev = (vals - own.unsqueeze(0)).norm(dim=1, keepdim=True)
            mag = vals.norm(dim=1, keepdim=True)
            feat = torch.cat([dev, mag, quality], dim=1)            # (k,5)
            w = torch.softmax(self.score(feat).squeeze(1), dim=0)   # (k,)
            agg = (w.unsqueeze(1) * vals).sum(dim=0)
            g = torch.sigmoid(self.own_gate)
            return g * own + (1.0 - g) * agg


class HeteroAggregator:
    """Het-aware aggregator. Declares needs_quality so the rollout passes the
    per-neighbor link-quality features alongside the values. Falls back to the
    median if torch / the checkpoint is unavailable."""
    needs_quality = True
    name = "hetero_gnn"

    def __init__(self, net: Any = None):
        self.net = net

    def __call__(self, own: np.ndarray, vals: List[np.ndarray],
                 quality: Optional[List[List[float]]] = None) -> np.ndarray:
        own = np.asarray(own, dtype=float).reshape(-1)
        if not vals:
            return own
        A = np.asarray(vals, dtype=float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        own_v = own[:A.shape[1]]
        if self.net is None or not _HAS_TORCH or quality is None:
            return np.median(A, axis=0)
        Q = np.asarray(quality, dtype=float)
        if Q.shape[0] != A.shape[0]:
            return np.median(A, axis=0)
        with torch.no_grad():
            out = self.net(torch.tensor(own_v, dtype=torch.float32),
                           torch.tensor(A, dtype=torch.float32),
                           torch.tensor(Q, dtype=torch.float32)).numpy()
        return out


def load_hetero_gnn() -> HeteroAggregator:
    path = os.path.join(MODEL_DIR, "hetero_gnn.pt")
    if not _HAS_TORCH or not os.path.exists(path):
        return HeteroAggregator(net=None)
    net = HeteroAttnNet()
    try:
        net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()
    except Exception:
        return HeteroAggregator(net=None)
    return HeteroAggregator(net=net)

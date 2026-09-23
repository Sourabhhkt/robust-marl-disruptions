"""
Milestone 10 WP2 training: heterogeneity-aware graph-attention aggregator.

Supervised: recover the honest neighborhood consensus target from a neighbor set
whose delivered values are corrupted in proportion to their *link quality* -- a
low-precision neighbor contributes a quantized (noisy) value and a high-age
neighbor a stale (drifted) value, while the quality features [age, reliability,
precision] are provided as input. The network learns to down-weight low-quality
neighbors, which a plain mean cannot, closing the Milestone 9 heterogeneity gap.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

import coordinators_m10 as C10

MODEL_DIR = C10.MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)


def sample_hetero_neighborhood(rng, max_k=8, spread=0.3):
    """(own, vals(k,1), quality(k,3), target). Honest neighbors near a common
    center; low-precision and high-age neighbors carry larger value error."""
    own = rng.uniform(-1.0, 1.0)
    k = int(rng.integers(2, max_k + 1))
    center = own + rng.normal(0.0, spread)
    vals, qual = [], []
    for _ in range(k):
        rel = rng.uniform(0.4, 1.0)
        pb = rng.choice([0, 2, 3, 4])            # 0 = full precision
        age = float(rng.integers(0, 6))
        err = 0.0
        if pb > 0:                                # quantization noise ~ step/2
            err += rng.uniform(-1, 1) * (3.0 / (2 ** pb))
        err += rng.normal(0.0, 0.06 * age)        # staleness drift
        vals.append(center + err)
        prec = 1.0 if pb <= 0 else min(1.0, pb / 8.0)
        qual.append([age / 10.0, rel, prec])
    return own, np.array(vals, dtype=np.float32).reshape(-1, 1), \
        np.array(qual, dtype=np.float32), float(center)


def train_hetero_gnn(epochs=4000, batch=64, seed=0, lr=2e-3, device="cpu"):
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = C10.HeteroAttnNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.MSELoss()
    t0 = time.time(); losses = []
    for ep in range(epochs):
        opt.zero_grad(); loss = 0.0
        for _ in range(batch):
            own, vals, qual, tgt = sample_hetero_neighborhood(rng)
            o = torch.tensor([own], dtype=torch.float32, device=dev)
            V = torch.tensor(vals, device=dev); Q = torch.tensor(qual, device=dev)
            loss = loss + lossf(net(o, V, Q), torch.tensor([tgt], device=dev))
        loss = loss / batch; loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (ep + 1) % 1000 == 0:
            print(f"  [hetero_gnn] epoch {ep+1:5d}  loss {np.mean(losses[-200:]):.5f}", flush=True)
    torch.save(net.state_dict(), os.path.join(MODEL_DIR, "hetero_gnn.pt"))
    # held-out: het-aware vs mean vs median vs deviation-only proxy
    net.eval(); herr, merr, medr = [], [], []
    for _ in range(4000):
        own, vals, qual, tgt = sample_hetero_neighborhood(rng)
        with torch.no_grad():
            g = float(net(torch.tensor([own], dtype=torch.float32, device=dev),
                          torch.tensor(vals, device=dev),
                          torch.tensor(qual, device=dev)).item())
        herr.append((g - tgt) ** 2); merr.append((float(vals.mean()) - tgt) ** 2)
        medr.append((float(np.median(vals)) - tgt) ** 2)
    stats = {"hetero_gnn_mse": float(np.mean(herr)), "mean_mse": float(np.mean(merr)),
             "median_mse": float(np.mean(medr)), "runtime_s": round(time.time() - t0, 1)}
    print(f"  [hetero_gnn] held-out MSE: het-aware {stats['hetero_gnn_mse']:.4f}  "
          f"mean {stats['mean_mse']:.4f}  median {stats['median_mse']:.4f}", flush=True)
    with open(os.path.join(MODEL_DIR, "hetero_gnn_training.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train_hetero_gnn(epochs=args.epochs, device=args.device)

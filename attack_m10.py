"""
Milestone 10 WP3a: worst-case adversarial attack search.

The Milestone 9 study used a finite set of attack families (Gaussian, constant
bias, sign-flip). This searches, per coordinator, for the linear Byzantine attack
payload = bias + slope * own_state that MAXIMIZES the team's final disagreement,
replacing the fixed attacks with an adversarial optimization. The result is a
worst-case robustness ranking: how badly each coordinator can be made to fail by
an adversary that tunes its message to that coordinator.

A coarse grid locates the basin and a short hill-climb refines it (the rollout is
not differentiable through the channel, so a gradient-free search is used).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np

import coordinators as C
import coordinated as CO
from core import NetworkFaultConfig


def _disagreement(strat, params, frac, seeds, n=20, steps=60):
    st = C.get_strategy(strat) if isinstance(strat, str) else strat
    vals = []
    for s in seeds:
        cfg = NetworkFaultConfig(seed=int(s), byzantine_comm_corrupt_prob=1.0, spoof_scale=2.0)
        rec = CO.coordinated_consensus_rollout(cfg, int(s), st, n=n, d=1, steps=steps,
                                               byzantine_frac=frac, byz_attack="param",
                                               byz_params=(float(params[0]), float(params[1])))
        vals.append(rec["final_disagreement"])
    return float(np.median(vals))


def worst_case(strat, frac=0.25, seeds=range(6),
               biases=(-4, -2, -1, 0, 1, 2, 4), slopes=(-2, -1, 0, 1, 2)):
    """Grid + hill-climb over (bias, slope) maximizing median disagreement."""
    best, best_p = -1.0, (0.0, 0.0)
    for b in biases:
        for sl in slopes:
            d = _disagreement(strat, (b, sl), frac, seeds)
            if d > best:
                best, best_p = d, (float(b), float(sl))
    # local refinement
    step = 0.5
    for _ in range(8):
        improved = False
        for db, ds in [(step, 0), (-step, 0), (0, step), (0, -step)]:
            p = (best_p[0] + db, best_p[1] + ds)
            d = _disagreement(strat, p, frac, seeds)
            if d > best:
                best, best_p, improved = d, p, True
        if not improved:
            step *= 0.5
    return best, best_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--strategies", nargs="+",
                    default=["mean", "median", "trimmed", "gnn", "reco", "hetero_gnn", "oracle"])
    ap.add_argument("--out", default="results_m10/wp3_worstcase.json")
    args = ap.parse_args()
    seeds = range(args.seeds)
    out = {}
    print(f"worst-case linear attack (byz fraction {args.frac}, {args.seeds} seeds):")
    print(f"{'strategy':14s} {'gauss(ref)':>12s} {'worst-case':>12s} {'(bias,slope)':>16s}")
    for strat in args.strategies:
        ref = _disagreement(strat, (0, 0), args.frac, seeds)  # ~ gauss-like baseline via param(0,0)=0 payload
        gauss = float(np.median([CO.coordinated_consensus_rollout(
            NetworkFaultConfig(seed=s, byzantine_comm_corrupt_prob=1.0, spoof_scale=2.0),
            s, C.get_strategy(strat), n=20, d=1, steps=60, byzantine_frac=args.frac,
            byz_attack="gauss")["final_disagreement"] for s in seeds]))
        wc, p = worst_case(strat, args.frac, seeds)
        out[strat] = {"gauss": gauss, "worst_case": wc, "params": p}
        print(f"{strat:14s} {gauss:12.4f} {wc:12.4f} {('('+format(p[0],'.1f')+','+format(p[1],'.1f')+')'):>16s}",
              flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    # worst-case ranking (lower is more robust)
    rank = sorted(out.items(), key=lambda kv: kv[1]["worst_case"])
    print("worst-case robustness ranking (best first):",
          ", ".join(f"{k}={v['worst_case']:.3f}" for k, v in rank))


if __name__ == "__main__":
    main()

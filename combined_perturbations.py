"""
Milestone 10 WP3b: combined (simultaneous) perturbations.

Milestone 8/9 swept one impairment family at a time. This applies *pairs* of
families together (their config modifications merged) and asks whether the
combined failure is super-additive -- worse than what the individual impairments
predict. The interaction metric is

    interaction = V_combined / max(V_fam1, V_fam2)

so > 1 flags a super-additive (worse-than-either) interaction, the regime a
single-family sweep cannot reveal.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

import coordinators as C
import coordinated as CO
from runner import apply_family
from core import NetworkFaultConfig


def merged_cfg(pairs: List[Tuple[str, float]], seed: int) -> Tuple[NetworkFaultConfig, float]:
    """Merge the config modifications of several (family, value) impairments."""
    kw: Dict[str, Any] = {"seed": int(seed)}
    byz = 0.0
    for fam, val in pairs:
        cfg, ov = apply_family(fam, val, dict(kw))
        kw = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        byz = max(byz, float(ov.get("byzantine_frac", 0.0)))
    return NetworkFaultConfig(**kw), byz


def _disag(strat, pairs, seeds, n=20, steps=60):
    st = C.get_strategy(strat)
    vals = []
    for s in seeds:
        cfg, byz = merged_cfg(pairs, s)
        rec = CO.coordinated_consensus_rollout(cfg, int(s), st, n=n, d=1, steps=steps,
                                               byzantine_frac=byz)
        vals.append(rec["final_disagreement"])
    return float(np.median(vals))


# representative paired families (moderate single-family levels)
PAIRS = [(("msg_drop", 0.3), ("latency", 4)),
         (("msg_drop", 0.3), ("bandwidth", 2)),
         (("bandwidth", 2), ("byzantine", 0.25)),
         (("latency", 4), ("byzantine", 0.25)),
         (("jam", 0.25), ("crash", 0.1))]


def sweep(strategies=("mean", "median", "gnn"), seeds=range(8)) -> pd.DataFrame:
    rows = []
    for strat in strategies:
        for (f1, v1), (f2, v2) in PAIRS:
            d1 = _disag(strat, [(f1, v1)], seeds)
            d2 = _disag(strat, [(f2, v2)], seeds)
            dc = _disag(strat, [(f1, v1), (f2, v2)], seeds)
            base = max(d1, d2, 1e-9)
            rows.append({"strategy": strat, "pair": f"{f1}+{f2}",
                         "v_fam1": d1, "v_fam2": d2, "v_combined": dc,
                         "interaction": dc / base, "super_additive": dc > base * 1.5})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--strategies", nargs="+", default=["mean", "median", "gnn"])
    ap.add_argument("--out", default="results_m10/wp3b_combined.csv")
    args = ap.parse_args()
    df = sweep(tuple(args.strategies), range(args.seeds))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df[["strategy", "pair", "v_combined", "interaction", "super_additive"]]
          .to_string(index=False))
    sa = df[df.super_additive]
    print(f"\nsuper-additive interactions (combined > 1.5x worst single): {len(sa)}/{len(df)}")

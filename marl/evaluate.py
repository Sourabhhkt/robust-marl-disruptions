"""
Evaluate learned MARL policies and scripted (classical-coordinator) policies on
the cooperative consensus task under the Milestone 8/9 disruption families, for an
apples-to-apples robustness comparison on one shared environment.

Scripted policies map the classical aggregators onto the discrete action space
(always pick a fixed target/gain), so "mean", "median", and "trimmed-mean" appear
as fixed-action baselines alongside the learned IPPO / MAPPO / QMIX policies. An
"adaptive_oracle" upper bound picks, each step, whichever aggregator yields the
lowest next-step disagreement given ground truth.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core import NetworkFaultConfig
import runner as R
from runner import apply_family
from .envs import make_env, _TARGETS, _GAINS


# ---- policies -------------------------------------------------------------
class ScriptedPolicy:
    """A fixed classical aggregator as a policy. For the continuous synthetic env
    the action is (blend, gain): mean -> blend 0, median -> blend 1. For the
    discrete ablation env it is the matching (target, gain) action index."""
    def __init__(self, target: str, env, gain: float = 0.35):
        self.continuous = getattr(env, "continuous", False)
        if self.continuous:
            import math
            blend_raw = 10.0 if target == "median" else -10.0     # sigmoid -> 1 / 0
            g = min(max(gain / getattr(env, "gmax", 0.6), 1e-3), 1 - 1e-3)
            self.vec = np.array([blend_raw, math.log(g / (1 - g))], dtype=np.float32)
        else:
            self.a = _TARGETS.index(target) * len(_GAINS) + 2

    def act(self, obs, state, explore=False):
        n = obs.shape[0]
        if self.continuous:
            return np.tile(self.vec, (n, 1)), {}
        return np.full(n, self.a, dtype=np.int64), {}


class LearnedPolicy:
    def __init__(self, algo_name: str, ckpt: str, env):
        from .algos import make_algo
        self.algo = make_algo(algo_name, env.obs_dim, env.n_actions, env.state_dim,
                              env.n_agents, device="cpu",
                              continuous=getattr(env, "continuous", False),
                              act_dim=getattr(env, "act_dim", 2))
        import torch
        self.algo.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def act(self, obs, state, explore=False):
        return self.algo.act(obs, state, explore=False)


def make_policy(spec: str, env, ckpt_dir: str = "marl/checkpoints"):
    """spec is 'mean'/'median'/'trimmed'/'own' (scripted) or
    'ippo:<tag>' / 'mappo:<tag>' / 'maddpg:<tag>' / 'qmix:<tag>' (learned)."""
    if spec in _TARGETS:
        return ScriptedPolicy(spec, env)
    algo, tag = spec.split(":", 1)
    return LearnedPolicy(algo, os.path.join(ckpt_dir, tag + ".pt"), env)


# ---- evaluation -----------------------------------------------------------
def run_policy(policy, env, seed, cfg, byz_frac):
    obs = env.reset(seed=seed, cfg=cfg, byz_frac=byz_frac)
    done, info = False, {}
    ret = 0.0
    while not done:
        a, _ = policy.act(obs, env.state(), explore=False)
        obs, r, done, info = env.step(a)
        ret += r
    if "V0" in info:
        return info.get("disagreement", 0.0) / max(info["V0"], 1e-8)
    if "reg0" in info:
        return info.get("regret", 0.0) / max(info["reg0"], 1e-6)
    return -ret


EVAL_FAMILIES = ["msg_drop", "latency", "bandwidth", "spoof", "byzantine"]


def sweep(policies: Dict[str, Any], families=EVAL_FAMILIES, seeds=range(10),
          env_kw=None, ckpt_dir="marl/checkpoints") -> pd.DataFrame:
    env = make_env("consensus", regime="clean", **(env_kw or {}))
    rows = []
    pols = {name: make_policy(name, env, ckpt_dir) for name in policies}
    for fam in families:
        for val in R.FAMILIES[fam]["values"]:
            for s in seeds:
                cfg, env_ov = apply_family(fam, val, {"seed": int(s)})
                byz = float(env_ov.get("byzantine_frac", 0.0))
                for name, pol in pols.items():
                    d = run_policy(pol, env, int(s), cfg, byz)
                    rows.append({"policy": name, "family": fam, "value": float(val),
                                 "seed": int(s), "final_disagreement": d})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+",
                    default=["mean", "median", "trimmed"],
                    help="scripted names and/or learned 'algo:tag' specs")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results_m10/marl_eval.csv")
    ap.add_argument("--ckpt-dir", default="marl/checkpoints")
    args = ap.parse_args()
    df = sweep({p: p for p in args.policies}, seeds=range(args.seeds), ckpt_dir=args.ckpt_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("byzantine@0.25 median final disagreement by policy:")
    b = df[(df.family == "byzantine") & (np.isclose(df.value, 0.25))]
    print(b.groupby("policy")["final_disagreement"].median().round(4).to_string())

"""
Milestone 10 WP6: cross-modality transfer.

Both synthetic Dec-POMDPs share the continuous action space (blend, gain): the
policy maps local communication statistics to an adaptive robust-aggregation
choice. This evaluates whether a policy trained on one modality (numeric
consensus) transfers zero-shot to the other (economic dispatch, a priced
optimization), and vice versa -- the cross-modality generalization question the
proposal raises. Observation dimensions differ (9 vs 10), so the foreign
observation is truncated / zero-padded to the policy's input width; this crude
alignment is part of the test (a transferable policy should rely on the shared
leading statistics: own value, mean/median of received, spread).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict
import numpy as np
import torch

from core import NetworkFaultConfig
from runner import apply_family
from marl.envs import make_env
from marl.algos import make_algo


class PaddedPolicy:
    """Wrap a trained policy so it accepts another env's observation width."""

    def __init__(self, algo_name: str, ckpt: str, train_obs_dim: int, n_agents: int,
                 act_dim: int = 2, state_dim_train: int = 0):
        # network input width is fixed by training; state input only matters for
        # the critic, which evaluation never uses.
        self.obs_dim = train_obs_dim
        self.algo = make_algo(algo_name, train_obs_dim, 12, max(state_dim_train, 1),
                              n_agents, continuous=True, act_dim=act_dim)
        self.algo.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def _fit(self, obs: np.ndarray) -> np.ndarray:
        n, d = obs.shape
        if d == self.obs_dim:
            return obs
        if d > self.obs_dim:
            return obs[:, : self.obs_dim]
        out = np.zeros((n, self.obs_dim), dtype=obs.dtype)
        out[:, :d] = obs
        return out

    def act(self, obs, state, explore=False):
        # actor-only path (the critic needs the training env's state width, which
        # the foreign env does not provide; evaluation never uses the critic)
        o = torch.as_tensor(self._fit(np.asarray(obs)), dtype=torch.float32)
        with torch.no_grad():
            a = self.algo.actor.dist(o).mean
        return a.numpy(), {}


def _metric(env, info, ret):
    if "V0" in info:
        return info.get("disagreement", 0.0) / max(info["V0"], 1e-8)
    if "reg0" in info:
        return info.get("regret", 0.0) / max(info["reg0"], 1e-6)
    return -ret


def run_eval(policy, env, fam, val, seeds):
    vals = []
    for s in seeds:
        cfg, ov = apply_family(fam, val, {"seed": int(s)})
        obs = env.reset(seed=int(s), cfg=cfg, byz_frac=float(ov.get("byzantine_frac", 0)))
        done, ret, info = False, 0.0, {}
        while not done:
            a, _ = policy.act(obs, env.state(), explore=False)
            obs, r, done, info = env.step(a)
            ret += r
        vals.append(_metric(env, info, ret))
    return float(np.median(vals))


def transfer_matrix(ckpt_dir="marl/checkpoints", algo="mappo", regime="disrupt",
                    seeds=range(10), train_seeds=range(5)):
    """2x2 matrix: train modality x eval modality, under moderate disruption."""
    envs = {"consensus": make_env("consensus", regime="clean"),
            "dispatch": make_env("dispatch", regime="clean")}
    obs_dims = {"consensus": 9, "dispatch": 10}
    state_dims = {k: e.state_dim for k, e in envs.items()}
    conds = [("byzantine", 0.25), ("msg_drop", 0.5)]
    out: Dict[str, Any] = {}
    for train_env in ("consensus", "dispatch"):
        pols = []
        for ts in train_seeds:
            ckpt = os.path.join(ckpt_dir, f"{algo}_{train_env}_{regime}_s{ts}.pt")
            if os.path.exists(ckpt):
                pols.append(PaddedPolicy(algo, ckpt, obs_dims[train_env],
                                         envs[train_env].n_agents,
                                         state_dim_train=state_dims[train_env]))
        if not pols:
            continue
        for eval_env_name, env in envs.items():
            for fam, val in conds:
                vals = [run_eval(p, env, fam, val, seeds) for p in pols]
                out[f"{train_env}->{eval_env_name}|{fam}{val}"] = float(np.median(vals))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_m10/wp6_transfer.json")
    args = ap.parse_args()
    m = transfer_matrix()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(m, f, indent=2)
    print("cross-modality transfer (median normalized objective, lower=better):")
    for k, v in sorted(m.items()):
        print(f"  {k:42s} {v:.4f}")

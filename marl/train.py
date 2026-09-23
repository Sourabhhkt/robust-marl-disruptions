"""
Trainer CLI for the Milestone 10 deep-MARL baselines.

Drives episodic rollouts for the on-policy (IPPO/MAPPO) and off-policy (QMIX)
algorithms over a shared cooperative environment, periodically evaluates the
greedy policy (reporting the true objective, the final team disagreement), and
saves a checkpoint plus a learning curve.

Usage:
    python -m marl.train --algo mappo --env consensus --regime clean --seed 0 \
        --steps 150000 --outdir marl/checkpoints
    python -m marl.train --algo qmix  --env consensus --regime disrupt --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np

from .envs import make_env
from .algos import make_algo, PPO, QMIX, MADDPG


def _norm_metric(info):
    """Env-appropriate normalized objective: disagreement/V0 (consensus) or
    regret/reg0 (dispatch), falling back to the raw episode signal otherwise."""
    if "V0" in info:
        return info.get("disagreement", 0.0) / max(info["V0"], 1e-8)
    if "reg0" in info:
        return info.get("regret", 0.0) / max(info["reg0"], 1e-6)
    return -info.get("reward", 0.0)   # MPE: lower is better -> negate return proxy


def evaluate(algo, env, n_episodes=20, base_seed=10_000):
    """Greedy evaluation: median normalized objective and mean episode return."""
    dis, rets = [], []
    for e in range(n_episodes):
        obs = env.reset(seed=base_seed + e)
        done, ret = False, 0.0
        info = {}
        while not done:
            a, _ = algo.act(obs, env.state(), explore=False)
            obs, r, done, info = env.step(a)
            ret += r
        dis.append(_norm_metric(info))
        rets.append(ret)
    return float(np.median(dis)), float(np.mean(rets))


def train(algo_name, env_name, regime="clean", seed=0, steps=150_000,
          episodes_per_update=16, eval_every=10_000, device="cpu",
          outdir="marl/checkpoints", env_kw=None, quiet=False):
    env = make_env(env_name, regime=regime, **(env_kw or {}))
    eval_env = make_env(env_name, regime=regime, **(env_kw or {}))
    algo = make_algo(algo_name, env.obs_dim, env.n_actions, env.state_dim,
                     env.n_agents, device=device, seed=seed,
                     continuous=getattr(env, "continuous", False),
                     act_dim=getattr(env, "act_dim", 2))
    is_ppo = isinstance(algo, PPO)
    rng = np.random.default_rng(seed)

    os.makedirs(outdir, exist_ok=True)
    tag = f"{algo_name}_{env_name}_{regime}_s{seed}"
    curve = []
    env_steps = 0
    t0 = time.time()
    next_eval = 0

    while env_steps < steps:
        if is_ppo:
            algo._ep = []
            for _ in range(episodes_per_update):
                s = int(rng.integers(1 << 30))
                obs = env.reset(seed=s)
                algo.start_episode()
                done = False
                while not done:
                    state = env.state()
                    a, extra = algo.act(obs, state, explore=True)
                    nobs, r, done, info = env.step(a)
                    algo.store(obs, state, a, extra, r, env.alive_mask())
                    obs = nobs
                    env_steps += 1
                algo.end_episode()
            algo.update()
        else:  # QMIX
            s = int(rng.integers(1 << 30))
            obs = env.reset(seed=s)
            done = False
            while not done:
                state = env.state()
                a, _ = algo.act(obs, state, explore=True)
                nobs, r, done, info = env.step(a)
                algo.store(obs, state, a, {}, r, env.alive_mask(),
                           nobs, env.state(), done)
                algo.update()
                obs = nobs
                env_steps += 1

        if env_steps >= next_eval:
            d, ret = evaluate(algo, eval_env)
            curve.append({"step": env_steps, "final_disagreement": d, "return": ret})
            if not quiet:
                print(f"[{tag}] step {env_steps:7d}  eval_disagreement {d:.3e}  "
                      f"return {ret:7.3f}  ({time.time()-t0:.0f}s)", flush=True)
            next_eval += eval_every

    # final eval + save
    d, ret = evaluate(algo, eval_env, n_episodes=40)
    curve.append({"step": env_steps, "final_disagreement": d, "return": ret})
    import torch
    torch.save(algo.state_dict(), os.path.join(outdir, tag + ".pt"))
    with open(os.path.join(outdir, tag + "_curve.json"), "w") as f:
        json.dump({"tag": tag, "algo": algo_name, "env": env_name, "regime": regime,
                   "seed": seed, "steps": env_steps, "runtime_s": round(time.time() - t0, 1),
                   "final_disagreement": d, "curve": curve}, f, indent=2)
    if not quiet:
        print(f"[{tag}] DONE  final_disagreement {d:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    return {"tag": tag, "final_disagreement": d, "return": ret}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ippo", "mappo", "maddpg", "qmix"])
    ap.add_argument("--env", default="consensus")
    ap.add_argument("--regime", default="clean", choices=["clean", "disrupt"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="marl/checkpoints")
    args = ap.parse_args()
    train(args.algo, args.env, regime=args.regime, seed=args.seed, steps=args.steps,
          device=args.device, outdir=args.outdir)

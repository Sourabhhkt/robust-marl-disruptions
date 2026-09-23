"""
WP1 results driver: evaluate the trained MARL policies against the scripted
classical-aggregator baselines on the cooperative consensus/dispatch tasks across
the disruption families (10 eval seeds, the M8/M9 standard), aggregating over the
training seeds, and produce the comparison CSV plus figures.

The headline figure is the per-family robustness comparison; a second figure shows
the *suite-average* normalized objective, where an adaptive learned policy should
beat any single fixed aggregator (the central WP1 claim).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import runner as R
from runner import apply_family
from .envs import make_env
from .evaluate import make_policy, run_policy, EVAL_FAMILIES

LABELS = {"mean": "mean (classical)", "median": "median (classical)",
          "trimmed": "trimmed (classical)", "ippo": "IPPO (learned)",
          "mappo": "MAPPO (learned)", "maddpg": "MADDPG (learned)"}
COLORS = {"mean": "#7f7f7f", "median": "#1f77b4", "trimmed": "#17becf",
          "ippo": "#2ca02c", "mappo": "#d62728", "maddpg": "#9467bd"}


def discover_checkpoints(ckpt_dir: str, env: str, regime: str) -> Dict[str, List[str]]:
    """Return {algo: [tags...]} for all training-seed checkpoints of (env, regime)."""
    out: Dict[str, List[str]] = {}
    for pt in sorted(glob.glob(os.path.join(ckpt_dir, f"*_{env}_{regime}_s*.pt"))):
        tag = os.path.basename(pt)[:-3]
        algo = tag.split("_")[0]
        if algo in ("ippo", "mappo", "maddpg"):
            out.setdefault(algo, []).append(tag)
    return out


def evaluate_all(env_name="consensus", regime="disrupt", ckpt_dir="marl/checkpoints",
                 eval_seeds=10, families=None) -> pd.DataFrame:
    families = families or EVAL_FAMILIES
    env = make_env(env_name, regime="clean")
    # build policy -> list of policy objects (scripted: one; learned: per training seed)
    specs: Dict[str, List] = {}
    for s in ("mean", "median"):
        specs[s] = [make_policy(s, env, ckpt_dir)]
    for algo, tags in discover_checkpoints(ckpt_dir, env_name, regime).items():
        specs[algo] = [make_policy(f"{algo}:{t}", env, ckpt_dir) for t in tags]
    rows = []
    for fam in families:
        for val in R.FAMILIES[fam]["values"]:
            for name, pols in specs.items():
                vals = []
                for pol in pols:
                    for s in range(eval_seeds):
                        cfg, ov = apply_family(fam, val, {"seed": int(s)})
                        vals.append(run_policy(pol, env, int(s), cfg,
                                               float(ov.get("byzantine_frac", 0.0))))
                rows.append({"policy": name, "family": fam, "value": float(val),
                             "metric_p50": float(np.median(vals)),
                             "metric_p25": float(np.percentile(vals, 25)),
                             "metric_p75": float(np.percentile(vals, 75))})
    return pd.DataFrame(rows)


def fig_family(df, fam, outpath, ylabel="final disagreement V(T)"):
    sub = df[df.family == fam].sort_values("value")
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for name in [p for p in LABELS if p in sub.policy.unique()]:
        s = sub[sub.policy == name]
        y = np.maximum(s["metric_p50"].to_numpy(), 1e-12)
        ax.plot(s["value"], y, marker="o", label=LABELS[name], color=COLORS.get(name))
        ax.fill_between(s["value"], np.maximum(s["metric_p25"], 1e-12),
                        np.maximum(s["metric_p75"], 1e-12), alpha=0.12, color=COLORS.get(name))
    ax.set_yscale("log"); ax.set_xlabel(R.FAMILIES[fam]["label"]); ax.set_ylabel(ylabel)
    ax.set_title(f"Learned vs classical under {fam}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def fig_suite_average(df, outpath):
    """Suite-average normalized objective per policy (geometric mean across all
    family/level points), the adaptivity headline."""
    g = df.groupby("policy")["metric_p50"].apply(
        lambda x: float(np.exp(np.mean(np.log(np.maximum(x, 1e-12)))))).sort_values()
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    names = [p for p in g.index]
    ax.bar(range(len(names)), g.values, color=[COLORS.get(n, "#333") for n in names])
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LABELS.get(n, n) for n in names],
                                                         rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("suite-average objective (geo-mean)")
    ax.set_title("Adaptivity: average over the disruption suite"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="consensus")
    ap.add_argument("--regime", default="disrupt")
    ap.add_argument("--ckpt-dir", default="marl/checkpoints")
    ap.add_argument("--outdir", default="results_m10")
    ap.add_argument("--figdir", default="figures_m10")
    ap.add_argument("--eval-seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.figdir, exist_ok=True)
    df = evaluate_all(args.env, args.regime, args.ckpt_dir, args.eval_seeds)
    df.to_csv(os.path.join(args.outdir, f"wp1_{args.env}_{args.regime}.csv"), index=False)
    for fam in EVAL_FAMILIES:
        fig_family(df, fam, os.path.join(args.figdir, f"wp1_{args.env}_{args.regime}_{fam}.png"))
    fig_suite_average(df, os.path.join(args.figdir, f"wp1_{args.env}_{args.regime}_suiteavg.png"))
    print("byzantine@0.25:")
    print(df[(df.family == "byzantine") & (np.isclose(df.value, 0.25))]
          [["policy", "metric_p50"]].to_string(index=False))

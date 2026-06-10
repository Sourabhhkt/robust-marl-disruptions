"""
Figure generation and control-theoretic analysis.

Produces:
  - degradation curves (median + interquartile band) per task/family,
  - algorithm-comparison curves (one line per algorithm),
  - rate-distortion curves (performance vs bits/message),
  - Lyapunov / convergence trajectories,
  - scalability curves (vs number of agents).

Operates on the aggregated summary DataFrame from ``runner.aggregate`` (and, for
trajectory/scalability plots, re-runs the synthetic envs directly).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figures are placed one-per-row at full column width (~3.3 in) in a two-column
# layout. We render at FIGSIZE with large fonts so that, after LaTeX scales the
# image to \columnwidth, the in-figure text is about the size of the 10 pt body
# text (displayed pt ~= rendered pt * 3.3 / FIGSIZE_width).
FIGSIZE = (5.4, 3.9)
plt.rcParams.update({
    "font.size": 13.5,
    "axes.titlesize": 11,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "figure.dpi": 150,
    "axes.formatter.useoffset": False,  # no "+offset" / scientific-offset tick labels
})

from core import NetworkFaultConfig
import synth_envs as SE
import baselines as B
from runner import FAMILIES


def _ensure(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def plot_curves(summary: pd.DataFrame, tasks: Sequence[str], family: str, metric: str,
                outpath: str, ylabel: Optional[str] = None, title: Optional[str] = None,
                logy: bool = False, labels: Optional[Dict[str, str]] = None):
    """Median + IQR band of ``metric`` vs the swept value, one line per task."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for task in tasks:
        sub = summary[(summary.task_name == task) & (summary.family == family)].sort_values("value")
        if sub.empty or f"{metric}_p50" not in sub:
            continue
        x = sub["value"].to_numpy()
        p50 = sub[f"{metric}_p50"].to_numpy()
        p25 = sub[f"{metric}_p25"].to_numpy()
        p75 = sub[f"{metric}_p75"].to_numpy()
        lab = (labels or {}).get(task, task)
        if logy:  # strictly-positive quantity on a log axis; floor to avoid log(0)
            p50 = np.maximum(p50, 1e-12); p25 = np.maximum(p25, 1e-12); p75 = np.maximum(p75, 1e-12)
        line, = ax.plot(x, p50, marker="o", label=lab)
        ax.fill_between(x, p25, p75, alpha=0.2, color=line.get_color())
    ax.set_xlabel(FAMILIES.get(family, {}).get("label", family))
    ax.set_ylabel(ylabel or metric)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    if len(tasks) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_rate_distortion(summary: pd.DataFrame, tasks: Sequence[str], outpath: str,
                         distortion: str = "final_disagreement", labels=None):
    """Distortion vs actual delivered bits/message (bandwidth sweep)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for task in tasks:
        sub = summary[(summary.task_name == task) & (summary.family == "bandwidth")].copy()
        if sub.empty:
            continue
        # x-axis: nominal bits (value); value 0 == full precision -> plot as 32
        sub["bits"] = sub["value"].replace(0.0, 32.0)
        sub = sub.sort_values("bits")
        lab = (labels or {}).get(task, task)
        y50 = np.maximum(sub[f"{distortion}_p50"].to_numpy(), 1e-12)
        y25 = np.maximum(sub[f"{distortion}_p25"].to_numpy(), 1e-12)
        y75 = np.maximum(sub[f"{distortion}_p75"].to_numpy(), 1e-12)
        line, = ax.plot(sub["bits"], y50, marker="s", label=lab)
        ax.fill_between(sub["bits"], y25, y75, alpha=0.2, color=line.get_color())
    ax.set_xlabel("bits per message (uncompressed shown as 32)")
    ax.set_ylabel(distortion.replace("_", " "))
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_title("Rate-distortion: performance vs communication budget")
    if len(tasks) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def make_rate_distortion_fig(outpath: str, seeds: Sequence[int],
                             init_offset: float = 0.317, bits=(1, 2, 3, 4, 8, 0),
                             n: int = 20, steps: int = 60):
    """Rate-distortion figure for consensus, plotting the consensus error (squared
    deviation of the agreed value from the ideal unquantized consensus) versus the
    per-message bit budget. An off-lattice initial offset is used so that no coarse
    grid sits exactly on the consensus target; together with the error-from-ideal
    distortion this yields a clean monotone curve (the disagreement metric can read
    zero when agents lock onto a common but biased quantization level)."""
    rows = []
    for b in bits:
        for s in seeds:
            cfg = NetworkFaultConfig(seed=int(s), bandwidth_bits=(None if b == 0 else int(b)),
                                     quant_clip=(-3.0, 3.0))
            rec = SE.linear_consensus_rollout(cfg, int(s), B.AGGREGATORS["mean"],
                                              n=n, d=1, steps=steps, init_offset=init_offset)
            rows.append({"value": float(b), "consensus_error": rec["consensus_error"]})
    df = pd.DataFrame(rows)
    agg = (df.groupby("value")["consensus_error"]
             .agg(consensus_error_p25=lambda x: np.nanpercentile(x, 25),
                  consensus_error_p50=lambda x: np.nanpercentile(x, 50),
                  consensus_error_p75=lambda x: np.nanpercentile(x, 75)).reset_index())
    agg["task_name"] = "consensus_mean"; agg["family"] = "bandwidth"
    return plot_rate_distortion(agg, ["consensus_mean"], outpath, distortion="consensus_error")


def plot_lyapunov_trajectories(task: str, conditions: List[Dict[str, Any]], outpath: str,
                               seeds: Sequence[int] = (0, 1, 2, 3, 4), title: Optional[str] = None):
    """V(t) trajectories (median over seeds) for several channel conditions."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    spec = B.TASKS[task]
    for cond in conditions:
        cfg_kw = dict(cond.get("cfg", {}))
        env_ov = dict(cond.get("env", {}))
        traces = []
        for s in seeds:
            cfg = NetworkFaultConfig(seed=int(s), **cfg_kw)
            kw = dict(spec["defaults"]); kw.update(env_ov)
            run = spec["run"]
            # call underlying rollout to access V
            rec = _rollout_with_V(task, run, cfg, s, kw)
            if rec is not None:
                traces.append(rec)
        if not traces:
            continue
        L = min(len(v) for v in traces)
        arr = np.array([v[:L] for v in traces])
        med = np.median(arr, axis=0)
        ax.plot(np.arange(L), np.maximum(med, 1e-12), label=cond.get("label", ""))
    ax.set_xlabel("step t")
    ax.set_ylabel("Lyapunov V(t)  (disagreement)")
    ax.set_yscale("log")
    ax.set_title(title or f"{task}: convergence under disruption")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _rollout_with_V(task: str, run, cfg, seed, kw):
    """Re-run a synthetic task and return its V(t) trajectory."""
    agg_name = kw.get("algo", "mean")
    try:
        if task.startswith(("consensus", "rendezvous", "flocking")):
            d = kw.get("d", 2 if not task.startswith("consensus") else 1)
            offsets = None
            rec = SE.linear_consensus_rollout(cfg, seed, B.AGGREGATORS[agg_name],
                                              n=kw.get("n", 20), d=d, steps=kw.get("steps", 60),
                                              task=task)
        elif task.startswith("formation"):
            n = kw.get("n", 12)
            rec = SE.linear_consensus_rollout(cfg, seed, B.AGGREGATORS[agg_name], n=n, d=2,
                                              offsets=SE._formation_offsets(n, 2.0),
                                              steps=kw.get("steps", 80), task="formation")
        elif task.startswith("dcop"):
            rec = SE.dcop_rollout(cfg, seed, algorithm=kw.get("algo", "minsum"),
                                  n=kw.get("n", 24), k_colors=kw.get("k_colors", 4),
                                  steps=kw.get("steps", 40), p_extra=kw.get("p_extra", 0.08))
        else:
            return None
        return np.asarray(rec["V"], dtype=float)
    except Exception:
        return None


def scalability_sweep(task: str, ns: Sequence[int], seeds: Sequence[int],
                      cfg_kw: Optional[Dict[str, Any]] = None,
                      extra_kw: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run a task across agent counts; return per-n median final error + convergence rate.

    ``extra_kw`` overrides env defaults (e.g. ``topology='ring'`` for a
    fixed-degree graph where algebraic connectivity decreases with n)."""
    spec = B.TASKS[task]
    rows = []
    for n in ns:
        for s in seeds:
            cfg = NetworkFaultConfig(seed=int(s), **(cfg_kw or {}))
            kw = dict(spec["defaults"]); kw["n"] = int(n)
            if extra_kw:
                kw.update(extra_kw)
            rec = spec["run"](cfg, s, **kw)
            rec["n_agents"] = int(n)
            rows.append({k: v for k, v in rec.items()
                         if isinstance(v, (int, float)) or k == "benchmark"})
    df = pd.DataFrame(rows)
    return df


def plot_scalability(df: pd.DataFrame, metric: str, outpath: str, ylabel: Optional[str] = None,
                     title: Optional[str] = None, logy: bool = False):
    g = df.groupby("n_agents")[metric]
    n = sorted(df["n_agents"].unique())
    p50 = np.array([np.nanpercentile(df[df.n_agents == x][metric], 50) for x in n])
    p25 = np.array([np.nanpercentile(df[df.n_agents == x][metric], 25) for x in n])
    p75 = np.array([np.nanpercentile(df[df.n_agents == x][metric], 75) for x in n])
    if logy:
        p50 = np.maximum(p50, 1e-12); p25 = np.maximum(p25, 1e-12); p75 = np.maximum(p75, 1e-12)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    line, = ax.plot(n, p50, marker="o")
    ax.fill_between(n, p25, p75, alpha=0.2, color=line.get_color())
    ax.set_xlabel("number of agents n")
    ax.set_ylabel(ylabel or metric)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_lambda2_vs_rate(summary: pd.DataFrame, task: str, outpath: str):
    """Empirical convergence rate rho vs effective algebraic connectivity (drop sweep)."""
    sub = summary[(summary.task_name == task) & (summary.family == "msg_drop")].sort_values("value")
    if sub.empty or "eff_lambda2_p50" not in sub:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(sub["eff_lambda2_p50"], sub["lyap_rho_p50"], marker="o")
    # stagger the drop= labels so neighbouring points (e.g. drop=0.0 and 0.1,
    # which nearly coincide) do not overlap.
    for i, (_, r) in enumerate(sub.iterrows()):
        dy = 8 if (i % 2 == 0) else -14
        ax.annotate(f"drop={r['value']:.1f}", (r["eff_lambda2_p50"], r["lyap_rho_p50"]),
                    fontsize=9, xytext=(4, dy), textcoords="offset points", ha="left")
    ax.set_xlabel(r"effective connectivity $\bar\lambda_2$")
    ax.set_ylabel(r"convergence rate $\rho$")
    ax.axhline(1.0, color="r", ls="--", lw=0.8, alpha=0.6)
    ax.set_title("Rate vs delivered connectivity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath

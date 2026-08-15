"""
Milestone 9 figure generation: comparative analysis across coordination
strategies, infrastructure domains, scale, and team heterogeneity.

Reuses ``analysis.plot_curves`` for per-family strategy comparisons (M9 task
names are ``base__strategy``, so a list of those names plots one line per
strategy) and adds the M9-specific figures here.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import FIGSIZE  # shared style
plt.rcParams.update({"font.size": 13.5, "axes.titlesize": 11, "axes.labelsize": 13.5,
                     "legend.fontsize": 11, "lines.linewidth": 2.4, "lines.markersize": 7})

# consistent strategy colors/labels across all figures
STRAT_LABEL = {"mean": "mean (M8 baseline)", "median": "median", "trimmed": "trimmed-mean",
               "oracle": "oracle", "gnn": "GNN (graph attn.)", "reco": "RECO",
               "hybrid": "hybrid MAS+DPS"}
STRAT_COLOR = {"mean": "#7f7f7f", "median": "#1f77b4", "trimmed": "#17becf",
               "oracle": "#2ca02c", "gnn": "#d62728", "reco": "#9467bd",
               "hybrid": "#ff7f0e"}


def _med_iqr(df, xcol, metric):
    xs = sorted(df[xcol].unique())
    p50 = np.array([np.nanpercentile(df[df[xcol] == x][metric], 50) for x in xs])
    p25 = np.array([np.nanpercentile(df[df[xcol] == x][metric], 25) for x in xs])
    p75 = np.array([np.nanpercentile(df[df[xcol] == x][metric], 75) for x in xs])
    return np.array(xs, float), p50, p25, p75


def plot_scalability_strategies(df: pd.DataFrame, metric: str, outpath: str,
                                strategies: Sequence[str], ylabel: str,
                                title: str, logy: bool = True):
    """One line per strategy: ``metric`` vs number of agents n."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s in strategies:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        x, p50, p25, p75 = _med_iqr(sub, "n_agents", metric)
        if logy:
            p50 = np.maximum(p50, 1e-12); p25 = np.maximum(p25, 1e-12); p75 = np.maximum(p75, 1e-12)
        ax.plot(x, p50, marker="o", label=STRAT_LABEL.get(s, s), color=STRAT_COLOR.get(s))
        ax.fill_between(x, p25, p75, alpha=0.15, color=STRAT_COLOR.get(s))
    ax.set_xlabel("number of agents n"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_hetero_sweep(df: pd.DataFrame, metric: str, outpath: str,
                      strategies: Sequence[str], ylabel: str, title: str,
                      logy: bool = True):
    """One line per strategy: ``metric`` vs heterogeneity level."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s in strategies:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        x, p50, p25, p75 = _med_iqr(sub, "hetero_level", metric)
        if logy:
            p50 = np.maximum(p50, 1e-12); p25 = np.maximum(p25, 1e-12); p75 = np.maximum(p75, 1e-12)
        ax.plot(x, p50, marker="o", label=STRAT_LABEL.get(s, s), color=STRAT_COLOR.get(s))
        ax.fill_between(x, p25, p75, alpha=0.15, color=STRAT_COLOR.get(s))
    ax.set_xlabel("team heterogeneity level"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_bandwidth_rd(df: pd.DataFrame, outpath: str, strategies: Sequence[str],
                      title: str = "Rate-distortion by strategy (consensus error)"):
    """Per-strategy rate-distortion: consensus error (deviation from the ideal
    unquantized consensus) vs bits per message. Uses the Milestone 8 metric, so
    no quantization-lattice-lock artifact."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s in strategies:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        x, p50, p25, p75 = _med_iqr(sub, "bits", "consensus_error")
        p50 = np.maximum(p50, 1e-12); p25 = np.maximum(p25, 1e-12); p75 = np.maximum(p75, 1e-12)
        ax.plot(x, p50, marker="s", label=STRAT_LABEL.get(s, s), color=STRAT_COLOR.get(s))
        ax.fill_between(x, p25, p75, alpha=0.15, color=STRAT_COLOR.get(s))
    ax.set_xlabel("bits per message (uncompressed shown as 32)")
    ax.set_ylabel("consensus error")
    ax.set_yscale("log"); ax.set_xscale("log", base=2)
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_infra_costgap(summary: pd.DataFrame, tasks: Sequence[str], family: str,
                       outpath: str, labels: Dict[str, str], title: str,
                       floor: float = 1e-3):
    """Infra dispatch cost gap vs a disruption family, with the cost gap floored
    at a small epsilon so the log-scale interquartile bands stay readable (the
    quantity touches numerical zero on a clean channel)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for task in tasks:
        sub = summary[(summary.task_name == task) & (summary.family == family)].sort_values("value")
        if sub.empty or "final_cost_gap_p50" not in sub:
            continue
        x = sub["value"].to_numpy()
        p50 = np.maximum(sub["final_cost_gap_p50"].to_numpy(), floor)
        p25 = np.maximum(sub["final_cost_gap_p25"].to_numpy(), floor)
        p75 = np.maximum(sub["final_cost_gap_p75"].to_numpy(), floor)
        strat = task.split("__")[1]
        line, = ax.plot(x, p50, marker="o", label=labels.get(task, task),
                        color=STRAT_COLOR.get(strat))
        ax.fill_between(x, p25, p75, alpha=0.15, color=line.get_color())
    ax.set_xlabel("message drop prob"); ax.set_ylabel(f"dispatch cost gap (floored at {floor:g})")
    ax.set_yscale("log"); ax.set_ylim(bottom=floor * 0.7)
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_infra_regret(summary: pd.DataFrame, tasks: Sequence[str], family: str,
                      outpath: str, labels: Dict[str, str], title: str,
                      floor: float = 1e-5):
    """Infra dispatch regret (feasibility-aware Lagrangian suboptimality, priced at
    the optimal shadow cost) vs a disruption family. Unlike the raw cost gap this
    is >= 0 and monotone in coordination error; the right-most (total-loss) point
    is the cost-blind equal-share fallback and is annotated."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xmax = None
    for task in tasks:
        sub = summary[(summary.task_name == task) & (summary.family == family)].sort_values("value")
        if sub.empty or "final_dispatch_regret_p50" not in sub:
            continue
        x = sub["value"].to_numpy()
        p50 = np.maximum(sub["final_dispatch_regret_p50"].to_numpy(), floor)
        p25 = np.maximum(sub["final_dispatch_regret_p25"].to_numpy(), floor)
        p75 = np.maximum(sub["final_dispatch_regret_p75"].to_numpy(), floor)
        strat = task.split("__")[1]
        line, = ax.plot(x, p50, marker="o", label=labels.get(task, task),
                        color=STRAT_COLOR.get(strat))
        ax.fill_between(x, p25, p75, alpha=0.15, color=line.get_color())
        xmax = float(x.max())
    if xmax is not None:
        ax.axvline(xmax, color="0.6", ls=":", lw=1.0)
        ax.annotate("total loss:\nequal-share\nfallback", xy=(xmax, ax.get_ylim()[1]),
                    xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                    fontsize=8, color="0.35")
    ax.set_xlabel("message drop prob"); ax.set_ylabel("dispatch regret")
    ax.set_yscale("log")
    ax.set_title(title); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_comm_efficiency(df: pd.DataFrame, outpath: str, strategies: Sequence[str],
                         title: str = "Communication volume vs coordination quality"):
    """Scatter: messages sent vs final disagreement, one point per strategy
    (median over seeds), at a fixed clean-channel operating point."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for s in strategies:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        msgs = np.nanmedian(sub["msgs_sent"])
        dis = max(np.nanmedian(sub["final_disagreement"]), 1e-12)
        ax.scatter(msgs, dis, s=90, color=STRAT_COLOR.get(s), zorder=3)
        ax.annotate(STRAT_LABEL.get(s, s), (msgs, dis), fontsize=9,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("messages sent per episode"); ax.set_ylabel("final disagreement V(T)")
    ax.set_yscale("log"); ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_strategy_bars(summary: pd.DataFrame, task_names: Dict[str, str], family: str,
                       value: float, metric: str, outpath: str, ylabel: str, title: str):
    """Grouped bar at a single severe operating point, one bar per strategy."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels, vals, colors = [], [], []
    for strat, tn in task_names.items():
        sub = summary[(summary.task_name == tn) & (summary.family == family) &
                      (np.isclose(summary.value, value))]
        if sub.empty or f"{metric}_p50" not in sub:
            continue
        labels.append(STRAT_LABEL.get(strat, strat))
        vals.append(max(float(sub[f"{metric}_p50"].iloc[0]), 1e-12))
        colors.append(STRAT_COLOR.get(strat))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel); ax.set_yscale("log"); ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_sample_efficiency(eff_json: str, outpath: str):
    """RECO reward-redistribution: learning curves (redistributed vs sparse) with
    the gradient-variance-reduction factor annotated."""
    if not os.path.exists(eff_json):
        return None
    with open(eff_json) as f:
        d = json.load(f)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for mode, color in [("redistributed", "#9467bd"), ("sparse", "#7f7f7f")]:
        c = d.get(f"curve_{mode}")
        if c:
            ax.plot(np.linspace(0, 1, len(c)) * d.get(f"episodes_to_90pct_{mode}", len(c) * 1.0)
                    if False else np.arange(len(c)), c, label=mode, color=color)
    vr = d.get("variance_reduction_x", None)
    ax.set_xlabel("training updates (smoothed)"); ax.set_ylabel("consensus quality")
    ttl = "RECO: reward redistribution vs sparse credit"
    if vr:
        ttl += f"\n(gradient-variance reduction {vr:.1f}x)"
    ax.set_title(ttl); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath


def plot_cross_domain_bars(summary: pd.DataFrame, domains: Sequence[str], strategy: str,
                           family: str, value: float, metric: str, outpath: str,
                           ylabel: str, title: str):
    """Bar of a physical-violation / cost metric per infrastructure domain at a
    fixed disruption level (one strategy)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels, vals = [], []
    for dom in domains:
        tn = f"ed_{dom}__{strategy}"
        sub = summary[(summary.task_name == tn) & (summary.family == family) &
                      (np.isclose(summary.value, value))]
        if sub.empty or f"{metric}_p50" not in sub:
            continue
        labels.append(dom); vals.append(float(sub[f"{metric}_p50"].iloc[0]))
    ax.bar(range(len(vals)), vals, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"][:len(vals)])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    return outpath

"""
Milestone 9 experiment driver: comparative evaluation of advanced coordination
strategies (oracle, graph-attention GNN, reward-redistribution RECO, hybrid
MAS+DPS) against the Milestone 8 classical baselines, across:

  1. Resilience  -- synthetic consensus under every disruption family.
  2. Cross-domain infrastructure -- distributed economic dispatch on electric
     power (pandapower) and gas (pandapipes) networks, plus the abstract sandbox.
  3. Scalability -- hybrid vs flat vs oracle as the team grows on a sparse ring.
  4. Adaptability across heterogeneous teams -- every strategy vs heterogeneity.
  5. Communication efficiency -- messages sent vs coordination quality.

Writes per-experiment and combined result CSVs, a reproducibility manifest, and
all figures. Reuses the Milestone 8 runner (perturbation families, two-stage
percentile aggregation) and figure helpers.

Usage:
    python run_m9.py --seeds 10 --workers 14 --outdir results_m9 --figdir figures_m9
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core import NetworkFaultConfig
import runner as R
from runner import apply_family, aggregate, save_results
from baselines_m9 import run_task_m9
import analysis as A
import analysis_m9 as A9

STRATEGIES = ["mean", "median", "trimmed", "oracle", "gnn", "reco", "hybrid"]


# ============================================================
# Multiprocessing worker (top-level, picklable: passes strategy by name)
# ============================================================
def _run_one_m9(args):
    (task_name, family, value, seed, episode, base_cfg_kwargs, base_env_overrides) = args
    cfg_kw = dict(base_cfg_kwargs or {}); cfg_kw["seed"] = int(seed)
    env_ov: Dict[str, Any] = {}
    if family == "hetero":
        cfg = NetworkFaultConfig(**cfg_kw)
        env_ov = {"hetero_level": float(value)}
    elif family == "clean":
        cfg = NetworkFaultConfig(**cfg_kw)
    else:
        cfg, env_ov = apply_family(family, value, cfg_kw)
    env_overrides = dict(base_env_overrides or {}); env_overrides.update(env_ov)
    try:
        rec = run_task_m9(task_name, cfg, int(seed), env_overrides)
    except Exception as e:
        rec = {"return": float("nan"), "error": repr(e)[:200]}
    rec = {k: v for k, v in rec.items()
           if isinstance(v, (int, float, str, bool)) or v is None}
    rec.update({"task_name": task_name, "family": family,
                "value": (float(value) if value is not None else 0.0),
                "seed": int(seed), "episode": int(episode)})
    return rec


def sweep_m9(jobs, workers):
    rows = []
    if workers <= 1:
        rows = [_run_one_m9(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_run_one_m9, jobs, chunksize=4):
                rows.append(r)
    return pd.DataFrame(rows)


# ============================================================
# Experiment 1: strategy resilience on synthetic consensus
# ============================================================
RESIL_FAMILIES = ["msg_drop", "latency", "bandwidth", "spoof", "jam", "byzantine"]


def exp_resilience(seeds, workers, base_kw=None):
    jobs = []
    for strat in STRATEGIES:
        tn = f"cons__{strat}"
        for fam in RESIL_FAMILIES:
            for val in R.FAMILIES[fam]["values"]:
                for s in seeds:
                    jobs.append((tn, fam, val, s, 0, base_kw, None))
    return sweep_m9(jobs, workers)


# ============================================================
# Experiment 1b: bandwidth rate-distortion (M8 consensus-error metric)
# ============================================================
def exp_bandwidth_rd(seeds, bits=(1, 2, 3, 4, 8, 0), init_offset=0.317,
                     strategies=("mean", "median", "trimmed", "gnn", "oracle")):
    """Per-strategy rate-distortion using the Milestone 8 consensus-error metric
    (squared deviation of the agreed value from the ideal unquantized consensus),
    with an off-lattice initial offset so no coarse grid sits on the target. This
    avoids the quantization-lattice-lock artifact that makes inter-agent
    disagreement a misleading distortion measure under quantization."""
    from baselines_m9 import run_consensus_m9
    import coordinators as C
    rows = []
    for strat in strategies:
        for b in bits:
            for s in seeds:
                cfg = NetworkFaultConfig(seed=int(s),
                                         bandwidth_bits=(None if b == 0 else int(b)),
                                         quant_clip=(-3.0, 3.0))
                st = C.get_strategy(strat)
                from coordinated import coordinated_consensus_rollout
                rec = coordinated_consensus_rollout(cfg, int(s), st, n=20, d=1, steps=60,
                                                    init_offset=init_offset)
                rows.append({"strategy": strat, "bits": float(b) if b else 32.0,
                             "consensus_error": rec["consensus_error"]})
    return pd.DataFrame(rows)


# ============================================================
# Experiment 1c: GNN off-distribution Byzantine attack
# ============================================================
def exp_gnn_offdistribution(seeds, frac=0.25):
    """Test whether the learned graph-attention robustness transfers to attack
    signals not seen in training. The GNN is trained on wide Gaussian Byzantine
    noise; here we also evaluate a constant-bias and a sign-flip attack and
    compare against the (distribution-free) median."""
    import coordinators as C
    from coordinated import coordinated_consensus_rollout
    rows = []
    for attack in ["gauss", "bias", "signflip"]:
        for strat in ["median", "gnn"]:
            st = C.get_strategy(strat)
            dis, err = [], []
            for s in seeds:
                cfg = NetworkFaultConfig(seed=int(s), byzantine_comm_corrupt_prob=1.0,
                                         spoof_scale=2.0)
                rec = coordinated_consensus_rollout(cfg, int(s), st, n=20, d=1, steps=60,
                                                    byzantine_frac=frac, byz_attack=attack)
                dis.append(rec["final_disagreement"])
                err.append(rec["consensus_error"])   # error from the true consensus
            rows.append({"attack": attack, "strategy": strat,
                         "final_disagreement": float(np.median(dis)),
                         "error_from_truth": float(np.median(err))})
    return pd.DataFrame(rows)


# ============================================================
# Experiment 2: cross-domain infrastructure dispatch
# ============================================================
INFRA_DOMAINS = ["abstract", "power", "gas"]
INFRA_STRATS = ["mean", "median", "gnn", "oracle", "hybrid"]
INFRA_FAMILIES = ["msg_drop", "latency", "bandwidth", "byzantine"]


def exp_infra(seeds, workers):
    jobs = []
    for dom in INFRA_DOMAINS:
        for strat in INFRA_STRATS:
            tn = f"ed_{dom}__{strat}"
            for fam in INFRA_FAMILIES:
                for val in R.FAMILIES[fam]["values"]:
                    for s in seeds:
                        jobs.append((tn, fam, val, s, 0, None, None))
    return sweep_m9(jobs, workers)


# ============================================================
# Experiment 3: scalability (hybrid vs flat vs oracle on a ring)
# ============================================================
def exp_scalability(seeds, ns=(10, 20, 40, 80, 160), strategies=("mean", "oracle", "hybrid", "gnn")):
    from baselines_m9 import run_consensus_m9
    rows = []
    for cond, cfg_kw in [("clean", {}), ("drop0.3", {"msg_drop_prob": 0.3})]:
        for strat in strategies:
            for n in ns:
                nclust = max(2, int(round(np.sqrt(n))))
                for s in seeds:
                    cfg = NetworkFaultConfig(seed=int(s), **cfg_kw)
                    # steps=60 matches the Milestone 8 scalability horizon, so the
                    # flat-consensus baseline reproduces the M8 numbers exactly.
                    rec = run_consensus_m9(cfg, int(s), strategy_name=strat, task="consensus",
                                           n=int(n), d=1, steps=60, topology="ring",
                                           p_extra=0.0, n_clusters=nclust)
                    rows.append({"strategy": strat, "n_agents": int(n), "cond": cond,
                                 "seed": int(s),
                                 "final_disagreement": rec["final_disagreement"],
                                 "lyap_rho": rec.get("lyap_rho", np.nan),
                                 "msgs_sent": rec.get("msgs_sent", np.nan)})
    return pd.DataFrame(rows)


# ============================================================
# Experiment 4: adaptability across heterogeneous teams
# ============================================================
def exp_hetero(seeds, workers, levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
    jobs = []
    # mild message loss so heterogeneity (per-agent reliability/precision) bites
    base = {"msg_drop_prob": 0.2}
    for strat in STRATEGIES:
        tn = f"cons__{strat}"
        for lv in levels:
            for s in seeds:
                jobs.append((tn, "hetero", lv, s, 0, base, None))
    df = sweep_m9(jobs, workers)
    df["hetero_level"] = df["value"]
    df["strategy"] = df["task_name"].str.split("__").str[1]
    return df


# ============================================================
# Figures
# ============================================================
def make_figures(summary, infra_summary, scal_df, hetero_df, clean_df, figdir,
                 bw_rd_df=None):
    os.makedirs(figdir, exist_ok=True)
    figs = []
    labels = {f"cons__{s}": A9.STRAT_LABEL[s] for s in STRATEGIES}

    # 1. Resilience: one figure per family, all strategies (disagreement metric).
    #    Bandwidth is handled separately below with the M8 consensus-error metric.
    for fam in ["byzantine", "msg_drop", "latency", "spoof", "jam"]:
        figs.append(A.plot_curves(summary, [f"cons__{s}" for s in STRATEGIES], fam,
                                  "final_disagreement",
                                  os.path.join(figdir, f"strat_{fam}.png"),
                                  ylabel="final disagreement V(T)", logy=True, labels=labels,
                                  title=f"Coordination strategies under {fam}"))

    # 1b. Bandwidth rate-distortion using the Milestone 8 consensus-error metric.
    if bw_rd_df is not None:
        figs.append(A9.plot_bandwidth_rd(bw_rd_df, os.path.join(figdir, "strat_bandwidth.png"),
                    ["mean", "median", "trimmed", "gnn", "oracle"]))

    # 2. Byzantine bar at fraction 0.25 (headline robustness ranking)
    figs.append(A9.plot_strategy_bars(summary, {s: f"cons__{s}" for s in STRATEGIES},
                "byzantine", 0.25, "final_disagreement",
                os.path.join(figdir, "strat_byzantine_bar.png"),
                "final disagreement V(T)", "Byzantine robustness at fraction 0.25"))

    # 3. Cross-domain infrastructure: dispatch regret (feasibility-aware, monotone)
    for dom in INFRA_DOMAINS:
        figs.append(A9.plot_infra_regret(infra_summary, [f"ed_{dom}__{s}" for s in INFRA_STRATS],
                    "msg_drop", os.path.join(figdir, f"infra_{dom}_regret_drop.png"),
                    {f"ed_{dom}__{s}": A9.STRAT_LABEL[s] for s in INFRA_STRATS},
                    f"Economic dispatch regret ({dom}) under message loss"))
    for dom in ["power", "gas"]:
        figs.append(A.plot_curves(infra_summary, [f"ed_{dom}__{s}" for s in INFRA_STRATS],
                    "byzantine", "phys_violation_frac", os.path.join(figdir, f"infra_{dom}_viol_byz.png"),
                    ylabel="physical violation fraction", logy=False,
                    labels={f"ed_{dom}__{s}": A9.STRAT_LABEL[s] for s in INFRA_STRATS},
                    title=f"{dom}: physical violations under Byzantine agents"))

    # 4. Scalability (clean + drop) per strategy
    for cond in ["clean", "drop0.3"]:
        sub = scal_df[scal_df["cond"] == cond]
        figs.append(A9.plot_scalability_strategies(sub, "final_disagreement",
                    os.path.join(figdir, f"scal_strat_{cond}.png"),
                    ["mean", "oracle", "hybrid", "gnn"], "final disagreement V(T)",
                    f"Scalability on a sparse ring ({cond})"))
        figs.append(A9.plot_scalability_strategies(sub, "lyap_rho",
                    os.path.join(figdir, f"scal_rho_{cond}.png"),
                    ["mean", "oracle", "hybrid", "gnn"], r"convergence rate $\rho$",
                    f"Convergence rate vs n ({cond})", logy=False))

    # 5. Heterogeneity adaptability
    figs.append(A9.plot_hetero_sweep(hetero_df, "final_disagreement",
                os.path.join(figdir, "hetero_disagreement.png"), STRATEGIES,
                "final disagreement V(T)", "Adaptability across heterogeneous teams"))

    # 6. Communication efficiency (clean consensus operating point)
    figs.append(A9.plot_comm_efficiency(clean_df, os.path.join(figdir, "comm_efficiency.png"),
                STRATEGIES))

    # 7. RECO sample efficiency (reward redistribution)
    eff = A9.plot_sample_efficiency(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "models", "reco_sample_efficiency.json"),
                                    os.path.join(figdir, "reco_sample_efficiency.png"))
    if eff:
        figs.append(eff)

    return [f for f in figs if f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--infra-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--outdir", type=str, default="results_m9")
    ap.add_argument("--figdir", type=str, default="figures_m9")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    iseeds = list(range(args.infra_seeds))
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    print("[1/5] strategy resilience (synthetic consensus)...", flush=True)
    resil = exp_resilience(seeds, args.workers)
    resil.to_csv(os.path.join(args.outdir, "m9_resilience_raw.csv"), index=False)
    resil_sum = aggregate(resil)
    bw_rd = exp_bandwidth_rd(seeds)
    bw_rd.to_csv(os.path.join(args.outdir, "m9_bandwidth_rd.csv"), index=False)
    offdist = exp_gnn_offdistribution(seeds)
    offdist.to_csv(os.path.join(args.outdir, "m9_gnn_offdistribution.csv"), index=False)
    print("    off-distribution attack (median vs gnn, frac 0.25):", flush=True)
    print(offdist.pivot(index="attack", columns="strategy",
                        values="final_disagreement").to_string(), flush=True)

    print("[2/5] cross-domain infrastructure dispatch...", flush=True)
    infra = exp_infra(iseeds, min(args.workers, 8))
    infra.to_csv(os.path.join(args.outdir, "m9_infra_raw.csv"), index=False)
    infra_sum = aggregate(infra)

    print("[3/5] scalability (hybrid vs flat vs oracle)...", flush=True)
    scal = exp_scalability(seeds)
    scal.to_csv(os.path.join(args.outdir, "m9_scalability.csv"), index=False)

    print("[4/5] adaptability across heterogeneous teams...", flush=True)
    hetero = exp_hetero(seeds, args.workers)
    hetero.to_csv(os.path.join(args.outdir, "m9_hetero.csv"), index=False)

    print("[5/5] figures...", flush=True)
    summary = pd.concat([resil_sum, infra_sum], ignore_index=True)
    save_results(args.outdir, "m9", pd.concat([resil, infra], ignore_index=True), summary,
                 manifest_extra={"seeds": seeds, "infra_seeds": iseeds,
                                 "strategies": STRATEGIES,
                                 "wall_clock_sec": round(time.time() - t0, 1)})
    clean_df = resil[(resil.family == "msg_drop") & (resil.value == 0.0)].copy()
    clean_df["strategy"] = clean_df["task_name"].str.split("__").str[1]
    figs = make_figures(resil_sum, infra_sum, scal, hetero, clean_df, args.figdir, bw_rd_df=bw_rd)
    print(f"done in {time.time()-t0:.1f}s; {len(figs)} figures in {args.figdir}/", flush=True)


if __name__ == "__main__":
    main()

"""
Experiment driver: run the full stress-test matrix, aggregate, save results
and manifest, and generate all figures.

Usage:
    python run_m8.py --seeds 10 --workers 12 --outdir results
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from runner import run_sweep, aggregate, save_results
import analysis as A


# Per-task perturbation families (only the meaningful ones for each benchmark).
COMM = ["msg_drop", "latency", "jitter", "bandwidth", "spoof", "jam", "partition"]
AGENT_SYNTH = ["crash", "sensor_noise", "byzantine"]
WRAP_FAMILIES = ["msg_drop", "latency", "bandwidth", "spoof", "crash", "sensor_noise", "actuator"]

TASK_FAMILIES = {
    # headline synthetic benchmarks: full coverage
    "consensus_mean":    COMM + AGENT_SYNTH,
    "consensus_median":  COMM + AGENT_SYNTH,
    "consensus_trimmed": ["msg_drop", "latency", "bandwidth", "spoof", "byzantine"],
    "rendezvous_mean":   ["msg_drop", "latency", "bandwidth", "crash", "byzantine"],
    "formation_mean":    ["msg_drop", "latency", "bandwidth", "crash", "byzantine"],
    "formation_median":  ["msg_drop", "byzantine"],
    "flocking_mean":     ["msg_drop", "latency", "bandwidth", "byzantine"],
    # DPS
    "dcop_minsum":       ["msg_drop", "latency", "bandwidth", "spoof", "crash", "byzantine"],
    "dcop_dsa":          ["msg_drop", "latency", "bandwidth", "spoof", "crash", "byzantine"],
    # real benchmarks (wrapper path)
    "wntr_rule":         ["msg_drop", "latency", "bandwidth", "crash", "sensor_noise", "actuator"],
    "wntr_random":       ["msg_drop", "latency", "bandwidth", "crash"],
    "mpe_speaker_listener": ["msg_drop", "latency", "bandwidth", "spoof", "crash", "actuator"],
    "mpe_simple_reference": ["msg_drop", "latency", "bandwidth", "spoof", "crash"],
    "mpe_simple_spread":    ["msg_drop", "latency", "bandwidth", "crash"],
}


def run_matrix(seeds, workers, outdir):
    t0 = time.time()
    os.makedirs(os.path.join(outdir, "per_task"), exist_ok=True)
    all_raw = []
    for task, families in TASK_FAMILIES.items():
        ts = time.time()
        # WNTR runs the EPANET C library; keep its worker count modest for stability.
        w = min(4, workers) if task.startswith("wntr") else workers
        try:
            df = run_sweep(task, families, seeds=seeds, episodes=1, workers=w)
            df.to_csv(os.path.join(outdir, "per_task", f"{task}.csv"), index=False)
            all_raw.append(df)
            print(f"  {task:24s} {len(df):4d} rows  ({time.time()-ts:5.1f}s)", flush=True)
        except Exception as e:
            print(f"  {task:24s} FAILED: {repr(e)[:120]}", flush=True)
    raw = pd.concat(all_raw, ignore_index=True)
    summary = aggregate(raw)
    paths = save_results(outdir, "m8", raw, summary,
                         manifest_extra={"seeds_used": list(map(int, seeds)),
                                         "wall_clock_sec": round(time.time() - t0, 1)})
    print(f"matrix done in {time.time()-t0:.1f}s -> {paths['summary']}")
    return raw, summary


def make_figures(summary, figdir, seeds):
    _ = os.makedirs(figdir, exist_ok=True)
    figs = []

    # 1. Degradation: consensus mean across all comm families (return = -disagreement)
    for fam in ["msg_drop", "latency", "bandwidth", "spoof", "jam", "partition", "crash", "sensor_noise"]:
        figs.append(A.plot_curves(summary, ["consensus_mean"], fam, "final_disagreement",
                                  os.path.join(figdir, f"consensus_{fam}.png"),
                                  ylabel="final disagreement V(T)", logy=True,
                                  title=f"Average consensus under {fam}"))

    # 2. Robustness comparison: mean vs median vs trimmed under Byzantine
    figs.append(A.plot_curves(summary, ["consensus_mean", "consensus_median", "consensus_trimmed"],
                              "byzantine", "final_disagreement",
                              os.path.join(figdir, "byzantine_robustness.png"),
                              ylabel="final disagreement V(T)", logy=True,
                              title="Byzantine robustness: mean vs robust aggregation",
                              labels={"consensus_mean": "mean", "consensus_median": "median",
                                      "consensus_trimmed": "trimmed-mean"}))

    # 3. Rate-distortion (consensus + dcop)
    figs.append(A.make_rate_distortion_fig(
        os.path.join(figdir, "rate_distortion_consensus.png"), seeds))
    figs.append(A.plot_rate_distortion(summary, ["dcop_minsum", "dcop_dsa"],
                                       os.path.join(figdir, "rate_distortion_dcop.png"),
                                       distortion="final_conflict_frac",
                                       labels={"dcop_minsum": "min-sum", "dcop_dsa": "DSA"}))

    # 4. Lyapunov trajectories under disruption
    figs.append(A.plot_lyapunov_trajectories(
        "consensus_mean",
        [dict(label="clean"),
         dict(label="latency=4", cfg=dict(base_latency_steps=4)),
         dict(label="drop=0.5", cfg=dict(msg_drop_prob=0.5)),
         dict(label="drop=1.0", cfg=dict(msg_drop_prob=1.0)),
         dict(label="2 bits", cfg=dict(bandwidth_bits=2, quant_clip=(-3, 3)))],
        os.path.join(figdir, "lyapunov_trajectories.png"), seeds=seeds))

    # 5. Consensus rate vs effective algebraic connectivity
    r = A.plot_lambda2_vs_rate(summary, "consensus_mean",
                               os.path.join(figdir, "rho_vs_lambda2.png"))
    if r:
        figs.append(r)

    # 6. DCOP degradation (conflict fraction) under drop + byzantine
    figs.append(A.plot_curves(summary, ["dcop_minsum", "dcop_dsa"], "msg_drop", "final_conflict_frac",
                              os.path.join(figdir, "dcop_drop.png"),
                              ylabel="final conflict fraction",
                              title="DCOP graph colouring under message loss",
                              labels={"dcop_minsum": "min-sum", "dcop_dsa": "DSA"}))
    figs.append(A.plot_curves(summary, ["dcop_minsum", "dcop_dsa"], "byzantine", "final_conflict_frac",
                              os.path.join(figdir, "dcop_byzantine.png"),
                              ylabel="final conflict fraction",
                              title="DCOP under Byzantine agents",
                              labels={"dcop_minsum": "min-sum", "dcop_dsa": "DSA"}))

    # 7. Real comm-dependent benchmark: MPE speaker-listener
    figs.append(A.plot_curves(summary, ["mpe_speaker_listener"], "msg_drop", "return",
                              os.path.join(figdir, "mpe_sl_drop.png"),
                              ylabel="episode return",
                              title="MPE speaker-listener: message drop"))
    figs.append(A.plot_curves(summary, ["mpe_speaker_listener"], "bandwidth", "return",
                              os.path.join(figdir, "mpe_sl_bandwidth.png"),
                              ylabel="episode return",
                              title="MPE speaker-listener: bandwidth (one-hot goal)"))
    figs.append(A.plot_curves(summary, ["mpe_speaker_listener"], "spoof", "return",
                              os.path.join(figdir, "mpe_sl_spoof.png"),
                              ylabel="episode return",
                              title="MPE speaker-listener: spoofing"))

    # 8. Silent-failure diagnostic: WNTR (and MPE spread) -- channel degrades but
    #    task return is flat because the message is not on the decision-critical path.
    figs.append(A.plot_curves(summary, ["wntr_rule"], "msg_drop", "comm_zero_frac",
                              os.path.join(figdir, "silent_wntr_zerofrac.png"),
                              ylabel="comm zero fraction",
                              title="WNTR: channel degradation is measured..."))
    figs.append(A.plot_curves(summary, ["wntr_rule"], "msg_drop", "return",
                              os.path.join(figdir, "silent_wntr_return.png"),
                              ylabel="episode return",
                              title="...but task return is unchanged (silent failure)"))
    figs.append(A.plot_curves(summary, ["mpe_simple_spread"], "msg_drop", "return",
                              os.path.join(figdir, "silent_spread_return.png"),
                              ylabel="episode return",
                              title="MPE spread (greedy coverage): comm-independent"))

    return [f for f in figs if f]


def make_scalability(figdir, seeds):
    figs = []
    # Fixed-degree ring: algebraic connectivity lambda2 ~ 1/n^2, so the
    # connectivity bottleneck is exposed as n grows.
    ring = dict(topology="ring", p_extra=0.0)
    df_clean = A.scalability_sweep("consensus_mean", ns=[10, 20, 40, 80, 160], seeds=seeds,
                                   extra_kw=ring)
    figs.append(A.plot_scalability(df_clean, "lyap_rho", os.path.join(figdir, "scal_rho_clean.png"),
                                   ylabel=r"convergence rate $\rho$",
                                   title="Consensus convergence rate vs n (ring)"))
    figs.append(A.plot_scalability(df_clean, "nominal_lambda2", os.path.join(figdir, "scal_lambda2.png"),
                                   ylabel=r"algebraic connectivity $\lambda_2$", logy=True,
                                   title="Connectivity vs n (fixed-degree ring)"))
    figs.append(A.plot_scalability(df_clean, "final_disagreement",
                                   os.path.join(figdir, "scal_tte_clean.png"),
                                   ylabel="final disagreement V(T)", logy=True,
                                   title="Final disagreement vs n (ring, clean)"))
    # under message loss
    df_drop = A.scalability_sweep("consensus_mean", ns=[10, 20, 40, 80, 160], seeds=seeds,
                                  cfg_kw=dict(msg_drop_prob=0.3), extra_kw=ring)
    figs.append(A.plot_scalability(df_drop, "final_disagreement",
                                   os.path.join(figdir, "scal_finaldis_drop.png"),
                                   ylabel="final disagreement V(T)", logy=True,
                                   title="Final disagreement vs n (ring, 30% loss)"))
    return [f for f in figs if f], df_clean, df_drop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--figdir", type=str, default="figures")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    raw, summary = run_matrix(seeds, args.workers, args.outdir)
    figs = make_figures(summary, args.figdir, seeds)
    scal_figs, df_clean, df_drop = make_scalability(args.figdir, seeds)
    df_clean.to_csv(os.path.join(args.outdir, "m8_scalability_clean.csv"), index=False)
    df_drop.to_csv(os.path.join(args.outdir, "m8_scalability_drop.csv"), index=False)
    print(f"generated {len(figs)+len(scal_figs)} figures in {args.figdir}/")

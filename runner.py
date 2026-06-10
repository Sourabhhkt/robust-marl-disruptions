"""
Experiment runner: perturbation sweeps with multi-seed percentile aggregation.

Two-stage aggregation protocol:
  - sweep one perturbation parameter at a time (others at baseline),
  - run multiple seeds (x episodes),
  - aggregate in two stages: mean within seed, then 25/50/75 percentiles across
    seeds (median curve + interquartile band),
  - write per-episode rows (CSV), an aggregated summary (CSV), and a JSON
    manifest (seeds, config, package versions, runtime) for reproducibility.

Parallelism: the (value x seed x episode) grid is embarrassingly parallel and
is dispatched over a process pool. Workers rebuild the config locally, so no
unpicklable callables (jam/partition functions) cross the process boundary.
"""
from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core import NetworkFaultConfig
from baselines import run_task


# ============================================================
# Perturbation families
# ============================================================
# Each family: name -> dict(values=[...], label="x-axis label").
# The mapping value -> config is done in apply_family (top-level, picklable path).
FAMILIES: Dict[str, Dict[str, Any]] = {
    "msg_drop":     dict(values=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], label="message drop prob"),
    "latency":      dict(values=[0, 1, 2, 4, 8], label="base latency (steps)"),
    "jitter":       dict(values=[0, 1, 2, 4, 8], label="jitter (steps)"),
    "bandwidth":    dict(values=[1, 2, 3, 4, 8, 0], label="bits/message (0=full)"),
    "spoof":        dict(values=[0.0, 0.1, 0.3, 0.5, 1.0], label="spoof prob"),
    "jam":          dict(values=[0.0, 0.1, 0.25, 0.5, 0.75], label="jammed fraction"),
    "partition":    dict(values=[0.0, 0.1, 0.25, 0.5], label="partitioned fraction"),
    "crash":        dict(values=[0.0, 0.01, 0.05, 0.1, 0.2], label="crash prob"),
    "sensor_noise": dict(values=[0.0, 0.1, 0.3, 0.5, 1.0], label="sensor noise scale"),
    "actuator":     dict(values=[0.0, 0.1, 0.3, 0.5], label="actuator fault prob"),
    "byzantine":    dict(values=[0.0, 0.1, 0.25, 0.5], label="byzantine fraction"),
}


def _first_frac_agents(frac: float):
    """Return a picklable-by-rebuild function selecting the first frac of agents."""
    def fn(t, agents):
        agents = list(agents)
        k = int(round(frac * len(agents)))
        return set(agents[:k])
    return fn


def apply_family(name: str, value: Any, base_cfg_kwargs: Dict[str, Any]):
    """Return (cfg, env_overrides) for one perturbation value of one family."""
    kw = dict(base_cfg_kwargs or {})
    env_overrides: Dict[str, Any] = {}

    if name == "msg_drop":
        kw["msg_drop_prob"] = float(value)
    elif name == "latency":
        kw["base_latency_steps"] = int(value)
    elif name == "jitter":
        kw["jitter_steps"] = int(value)
    elif name == "bandwidth":
        kw["bandwidth_bits"] = (None if int(value) == 0 else int(value))
        kw.setdefault("quant_clip", (-3.0, 3.0))
    elif name == "spoof":
        kw["spoof_prob"] = float(value)
        kw.setdefault("spoof_scale", 1.0)
    elif name == "jam":
        kw["jammed_agents_fn"] = _first_frac_agents(float(value))
        kw["jam_drop_prob"] = 1.0
    elif name == "partition":
        kw["partitioned_agents_fn"] = _first_frac_agents(float(value))
    elif name == "crash":
        kw["crash_prob"] = float(value)
        kw.setdefault("crash_duration", 5)
    elif name == "sensor_noise":
        kw["sensor_noise_prob"] = 1.0
        kw["sensor_noise_scale"] = float(value)
    elif name == "actuator":
        kw["actuator_fault_prob"] = float(value)
        kw.setdefault("actuator_fault_mode", "random")
    elif name == "byzantine":
        env_overrides["byzantine_frac"] = float(value)
        kw["byzantine_comm_corrupt_prob"] = 1.0
        kw.setdefault("spoof_scale", 2.0)
    else:
        raise KeyError(f"unknown family {name}")

    cfg = NetworkFaultConfig(**kw)
    return cfg, env_overrides


# ============================================================
# Worker
# ============================================================
def _run_one(args):
    (task_name, family, value, seed, episode,
     base_cfg_kwargs, base_env_overrides) = args
    cfg_kw = dict(base_cfg_kwargs or {})
    cfg_kw["seed"] = int(seed)
    cfg, env_ov = apply_family(family, value, cfg_kw)
    env_overrides = dict(base_env_overrides or {})
    env_overrides.update(env_ov)
    try:
        rec = run_task(task_name, cfg, seed=int(seed), env_overrides=env_overrides)
    except Exception as e:  # keep the sweep alive; mark the failure
        rec = {"return": float("nan"), "error": repr(e)[:200]}
    rec = {k: v for k, v in rec.items()
           if isinstance(v, (int, float, str, bool)) or v is None}
    rec.update({"task_name": task_name, "family": family, "value": (float(value) if value is not None else 0.0),
                "seed": int(seed), "episode": int(episode)})
    return rec


# ============================================================
# Sweep + aggregation
# ============================================================
def run_sweep(
    task_name: str,
    families: Sequence[str],
    seeds: Sequence[int],
    episodes: int = 1,
    base_cfg_kwargs: Optional[Dict[str, Any]] = None,
    base_env_overrides: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    jobs = []
    for family in families:
        for value in FAMILIES[family]["values"]:
            for seed in seeds:
                for ep in range(episodes):
                    jobs.append((task_name, family, value, seed, ep,
                                 base_cfg_kwargs, base_env_overrides))

    rows: List[Dict[str, Any]] = []
    workers = workers or max(1, min(14, (os.cpu_count() or 2) - 2))
    if workers <= 1:
        rows = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_run_one, jobs, chunksize=4):
                rows.append(r)
    return pd.DataFrame(rows)


# metrics to summarize (present-if-available)
def _numeric_metric_cols(df: pd.DataFrame) -> List[str]:
    skip = {"seed", "episode", "value"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Two-stage aggregation: mean within seed, then percentiles across seeds."""
    metric_cols = _numeric_metric_cols(df)
    group_keys = ["task_name", "family", "value"]
    # stage 1: mean within seed
    per_seed = (df.groupby(group_keys + ["seed"])[metric_cols]
                  .mean().reset_index())
    # stage 2: percentiles across seeds
    import warnings
    out_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for keys, g in per_seed.groupby(group_keys):
            row = dict(zip(group_keys, keys))
            row["n_seeds"] = g["seed"].nunique()
            for m in metric_cols:
                col = g[m].to_numpy(dtype=float)
                if np.all(np.isnan(col)):
                    row[f"{m}_p25"] = row[f"{m}_p50"] = row[f"{m}_p75"] = row[f"{m}_mean"] = float("nan")
                    continue
                row[f"{m}_p25"] = float(np.nanpercentile(col, 25))
                row[f"{m}_p50"] = float(np.nanpercentile(col, 50))
                row[f"{m}_p75"] = float(np.nanpercentile(col, 75))
                row[f"{m}_mean"] = float(np.nanmean(col))
            out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(group_keys).reset_index(drop=True)


# ============================================================
# Persistence + manifest
# ============================================================
def package_versions() -> Dict[str, str]:
    vers = {"python": platform.python_version()}
    for mod in ["numpy", "pandas", "scipy", "networkx", "torch", "pettingzoo", "mpe2", "wntr", "gymnasium"]:
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            vers[mod] = "absent"
    return vers


def save_results(outdir: str, name: str, raw: pd.DataFrame, summary: pd.DataFrame,
                 manifest_extra: Optional[Dict[str, Any]] = None):
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, f"{name}_raw.csv")
    sum_path = os.path.join(outdir, f"{name}_summary.csv")
    man_path = os.path.join(outdir, f"{name}_manifest.json")
    raw.to_csv(raw_path, index=False)
    summary.to_csv(sum_path, index=False)
    manifest = {
        "name": name,
        "packages": package_versions(),
        "n_raw_rows": int(len(raw)),
        "families": sorted(raw["family"].unique().tolist()) if "family" in raw else [],
        "tasks": sorted(raw["task_name"].unique().tolist()) if "task_name" in raw else [],
        "seeds": sorted(raw["seed"].unique().tolist()) if "seed" in raw else [],
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return {"raw": raw_path, "summary": sum_path, "manifest": man_path}


if __name__ == "__main__":
    # quick self-check
    df = run_sweep("consensus_mean", ["msg_drop"], seeds=[0, 1, 2], workers=4)
    print(aggregate(df)[["family", "value", "final_disagreement_p50", "delivery_rate_p50"]])

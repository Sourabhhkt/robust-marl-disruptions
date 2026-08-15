"""
Milestone 9 task registry and dispatch.

Tasks are addressed by a ``family__strategy`` name so that the multiprocessing
runner can pass plain strings across the process boundary (learned coordinators
are reconstructed from their name inside each worker, loading the checkpoint
once). Three task families:

  - ``cons__<strategy>``    : synthetic consensus under <strategy>
  - ``rendez__<strategy>``  : 2-D rendezvous under <strategy>
  - ``ed_<domain>__<strategy>`` : distributed economic dispatch on a physical
                                  network (<domain> in abstract/power/gas)

<strategy> is one of mean, median, trimmed, oracle, gnn, reco, hybrid. Records
are flattened to the same scalar schema as Milestone 8 (control-theoretic scalars
included) so ``runner.aggregate`` and the analysis code consume them unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from core import NetworkFaultConfig
import baselines as B
import coordinators as C
import coordinated as CO
import infra_envs as IE
from hetero import make_hetero_profile, homogeneous_profile


# strategy -> infra combiner (string for classical, learned-aggregator .combine)
def _infra_combiner(strategy_name: str):
    if strategy_name in ("mean", "median", "trimmed"):
        return strategy_name
    if strategy_name == "oracle":
        return "oracle"
    if strategy_name == "hybrid":
        return "mean"          # hybrid reduces to structured mean for the scalar ED
    st = C.get_strategy(strategy_name)         # gnn / reco learned aggregator
    return st.combine if hasattr(st, "combine") else "median"


def run_consensus_m9(cfg, seed, strategy_name="mean", task="consensus",
                     hetero_level=0.0, **kw) -> Dict[str, Any]:
    st = C.get_strategy(strategy_name)
    n = int(kw.get("n", 20))
    prof = (make_hetero_profile(n, seed, level=float(hetero_level))
            if hetero_level > 0 else None)
    d = int(kw.get("d", 1 if task == "consensus" else 2))
    rec = CO.coordinated_consensus_rollout(
        cfg, seed, st, n=n, d=d, steps=int(kw.get("steps", 60)),
        topology=kw.get("topology", "ring_plus"), p_extra=kw.get("p_extra", 0.15),
        byzantine_frac=float(kw.get("byzantine_frac", 0.0)),
        byz_attack=kw.get("byz_attack", "gauss"),
        init_offset=float(kw.get("init_offset", 0.0)), task=task,
        hetero_profile=prof, n_clusters=kw.get("n_clusters"),
        event_trigger=float(kw.get("event_trigger", 0.0)))
    out = B._flatten_synth(rec, strategy_name)
    out["strategy"] = strategy_name
    out["msgs_sent"] = rec.get("msgs_sent", np.nan)
    return out


def run_infra_m9(cfg, seed, strategy_name="mean", domain="abstract",
                 hetero_level=0.0, **kw) -> Dict[str, Any]:
    n = int(kw.get("n", 6))
    rec = IE.dispatch_consensus_rollout(
        cfg, seed, _infra_combiner(strategy_name), domain=domain, n=n,
        steps=int(kw.get("steps", 70)), byzantine_frac=float(kw.get("byzantine_frac", 0.0)),
        hetero=float(kw.get("hetero", 1.0)),
        physics=bool(kw.get("physics", True)))
    out = B._flatten_synth(rec, strategy_name)
    out["strategy"] = strategy_name
    out["domain"] = domain
    for k in ("final_power_mismatch", "final_cost_gap", "final_dispatch_regret",
              "phys_violation_frac", "phys_infeasible", "lambda_opt"):
        if k in rec:
            out[k] = rec[k]
    return out


# parameter defaults per task family
_DEFAULTS = {
    "cons":   dict(n=20, d=1, steps=60, topology="ring_plus"),
    "rendez": dict(n=20, d=2, steps=60, topology="ring_plus"),
}


def run_task_m9(task_name: str, cfg: NetworkFaultConfig, seed: int,
                env_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch a ``family__strategy`` task name."""
    base, strat = task_name.split("__", 1)
    kw = dict(env_overrides or {})
    if base.startswith("ed_"):
        domain = base[3:]
        rec = run_infra_m9(cfg, seed, strategy_name=strat, domain=domain, **kw)
    elif base in ("cons", "rendez"):
        d = dict(_DEFAULTS[base]); d.update(kw)
        task = "consensus" if base == "cons" else "rendezvous"
        rec = run_consensus_m9(cfg, seed, strategy_name=strat, task=task, **d)
    else:
        raise KeyError(f"unknown M9 task base {base}")
    rec.setdefault("task", task_name)
    rec["task_name"] = task_name
    return rec

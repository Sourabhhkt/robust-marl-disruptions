"""
Baseline and reference algorithms + a unified task registry.

A *task* is ``(name) -> run(cfg, seed, **env_overrides) -> record`` where the
record is a flat dict of scalars/strings ready for a CSV row. Synthetic-env
tasks additionally compute control-theoretic scalars (Lyapunov descent,
convergence rate, algebraic connectivity, etc.) inline.

Algorithm coverage (baseline algorithms):
  - consensus:   average / median / trimmed-mean aggregation
  - rendezvous:  2-D average consensus
  - formation:   relative-position formation control (mean / median)
  - flocking:    velocity-alignment consensus (scalability)
  - dcop:        DSA and min-sum (max-sum family) graph colouring
  - mpe:         scripted communication-dependent policies (speaker_listener,
                 simple_reference) + greedy policy (simple_spread)
  - wntr:        rule-based pressure controller
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from core import NetworkFaultConfig
import metrics as M
import synth_envs as SE


# ============================================================
# Consensus aggregators (the algorithm under test)
# ============================================================
def agg_mean(own: np.ndarray, vals: List[np.ndarray]) -> np.ndarray:
    return np.mean(vals, axis=0)


def agg_median(own: np.ndarray, vals: List[np.ndarray]) -> np.ndarray:
    return np.median(vals, axis=0)


def agg_trimmed(own: np.ndarray, vals: List[np.ndarray], frac: float = 0.25) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if arr.shape[0] <= 2:
        return np.mean(arr, axis=0)
    k = int(np.floor(frac * arr.shape[0]))
    out = np.empty(arr.shape[1])
    for c in range(arr.shape[1]):
        col = np.sort(arr[:, c])
        col = col[k: arr.shape[0] - k] if (arr.shape[0] - 2 * k) > 0 else col
        out[c] = np.mean(col)
    return out


AGGREGATORS: Dict[str, Callable] = {
    "mean": agg_mean,
    "median": agg_median,
    "trimmed": partial(agg_trimmed, frac=0.25),
}


# ============================================================
# Control-theoretic scalars from a synthetic-env record
# ============================================================
def _control_scalars(rec: Dict[str, Any], eps_frac: float = 0.01) -> Dict[str, float]:
    V = np.asarray(rec.get("V"), dtype=float)
    out: Dict[str, float] = {}
    if V.size >= 2:
        cr = M.convergence_rate(V)
        out["lyap_rho"] = cr["rho"]
        out["lyap_rho_r2"] = cr["r2"]
        out["lyap_monotone_frac"] = M.monotone_decrease_fraction(V)
        out["lyap_ultimate_bound"] = M.ultimate_bound(V)
        out["lyap_mean_drift"] = M.mean_drift(V)
        eps = max(1e-9, eps_frac * float(V[0]))
        tte = M.time_to_threshold(V, eps)
        out["time_to_eps"] = float(tte) if tte is not None else float("nan")
        out["converged"] = 1.0 if tte is not None else 0.0
    adj = rec.get("adj")
    if adj is not None and np.size(adj):
        out["nominal_lambda2"] = M.laplacian_lambda2(adj)
    eff = rec.get("eff_adjs")
    if eff:
        out["eff_lambda2"] = M.effective_lambda2_series(eff)
    return out


def _flatten_synth(rec: Dict[str, Any], algo: str) -> Dict[str, Any]:
    keep_scalar = {k: v for k, v in rec.items()
                   if isinstance(v, (int, float, str)) and k not in ("V",)}
    keep_scalar["algo"] = algo
    keep_scalar.update(_control_scalars(rec))
    return keep_scalar


# ============================================================
# Synthetic-env task runners
# ============================================================
def run_consensus(cfg, seed, algo="mean", d=1, **kw):
    rec = SE.linear_consensus_rollout(cfg, seed, AGGREGATORS[algo], d=d, task="consensus", **kw)
    return _flatten_synth(rec, algo)


def run_rendezvous(cfg, seed, algo="mean", **kw):
    rec = SE.linear_consensus_rollout(cfg, seed, AGGREGATORS[algo], d=2, task="rendezvous", **kw)
    return _flatten_synth(rec, algo)


def run_formation(cfg, seed, algo="mean", n=12, **kw):
    offsets = SE._formation_offsets(n, radius=2.0)
    rec = SE.linear_consensus_rollout(cfg, seed, AGGREGATORS[algo], n=n, d=2,
                                      offsets=offsets, task="formation", **kw)
    return _flatten_synth(rec, algo)


def run_flocking(cfg, seed, algo="mean", n=40, **kw):
    rec = SE.linear_consensus_rollout(cfg, seed, AGGREGATORS[algo], n=n, d=2,
                                      init_scale=1.0, task="flocking", **kw)
    return _flatten_synth(rec, algo)


def run_dcop(cfg, seed, algo="minsum", **kw):
    rec = SE.dcop_rollout(cfg, seed, algorithm=algo, **kw)
    return _flatten_synth(rec, algo)


# ============================================================
# Wrapped real-benchmark tasks (MPE, WNTR)
# ============================================================
def _run_wrapped(env, adapter, cfg, seed, policy, max_steps=50, routing_fn=None):
    from wrapper import InstrumentedUniversalCommFaultWrapper
    w = InstrumentedUniversalCommFaultWrapper(env, adapter, cfg, routing_fn=routing_fn)
    obs, _ = w.reset(seed=seed)
    ep_ret = 0.0
    steps = 0
    while w.agents and steps < max_steps:
        actions = policy(w, obs)
        obs, rewards, terms, truncs, _ = w.step(actions)
        ep_ret += float(np.mean(list(rewards.values()))) if rewards else 0.0
        steps += 1
        if (terms and all(terms.values())) or (truncs and all(truncs.values())):
            break
    rec = {"return": ep_ret, "steps": steps}
    rec.update(w.metrics())
    try:
        w.close()
    except Exception:
        pass
    return rec


# ---- WNTR rule-based controller --------------------------------------------
def run_wntr(cfg, seed, algo="rule_based", inp="Net3.inp", horizon=24, **kw):
    from env_shims import WNTRParallelEnv
    from adapter import ObsSliceCommAdapter

    env = WNTRParallelEnv(inp, horizon_steps=horizon)

    def slice_fn(obs):
        a = np.asarray(obs, float).reshape(-1); k = min(8, a.shape[0]); return a[-k:]

    def replace_fn(obs, msg):
        a = np.asarray(obs, float).copy().reshape(-1); k = min(msg.shape[0], a.shape[0]); a[-k:] = msg[:k]; return a

    adapter = ObsSliceCommAdapter(agents_list_fn=lambda e: e.possible_agents,
                                  slice_fn=slice_fn, replace_fn=replace_fn)

    def policy(w, obs):
        # rule-based: keep pumps/valves ON (open) to maintain pressure;
        # the random baseline toggles. Communication carries the shared global
        # feature tail that the rule consults.
        if algo == "random":
            return {a: w.action_space(a).sample() for a in w.agents}
        return {a: 1 for a in w.agents}

    rec = _run_wrapped(env, adapter, cfg, seed, policy, max_steps=horizon)
    rec["benchmark"] = "wntr"; rec["algo"] = algo; rec["n_agents"] = len(env.possible_agents)
    return rec


# ---- MPE scripted policies --------------------------------------------------
def _mpe_make(env_name, **kw):
    import importlib
    name_map = {
        "speaker_listener": "simple_speaker_listener_v4",
        "simple_reference": "simple_reference_v3",
        "simple_spread": "simple_spread_v3",
    }
    mod = importlib.import_module(f"mpe2.{name_map[env_name]}")
    return mod.parallel_env(**kw)


class _PZParallelShim:
    """Adapt a PettingZoo parallel_env to the wrapper's dict API."""
    def __init__(self, pz_env):
        self.env = pz_env

    @property
    def agents(self):
        return list(self.env.agents)

    @property
    def possible_agents(self):
        return list(self.env.possible_agents)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        return self.env.close()


def run_mpe(cfg, seed, env_name="speaker_listener", algo="scripted", max_cycles=25, **kw):
    from adapter import ObsSliceCommAdapter

    pz = _mpe_make(env_name, max_cycles=max_cycles)
    env = _PZParallelShim(pz)

    # comm = last `mdim` entries of each obs (obs-slice communication model)
    mdim = 3

    def slice_fn(obs):
        a = np.asarray(obs, float).reshape(-1); k = min(mdim, a.shape[0]); return a[-k:]

    def replace_fn(obs, msg):
        a = np.asarray(obs, float).copy().reshape(-1); k = min(msg.shape[0], a.shape[0]); a[-k:] = msg[:k]; return a

    adapter = ObsSliceCommAdapter(agents_list_fn=lambda e: e.possible_agents,
                                  slice_fn=slice_fn, replace_fn=replace_fn)

    def _move_toward(rel):
        # MPE discrete actions: 0=noop, 1=left(-x), 2=right(+x), 3=down(-y), 4=up(+y)
        dx, dy = float(rel[0]), float(rel[1])
        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1
        return 4 if dy > 0 else 3

    def scripted_speaker_listener(w, obs):
        # Listener must reach the landmark named by the speaker's message. The
        # message (goal one-hot, dim 3) is delivered into the listener's comm slot
        # (last 3 obs entries) by the channel; corrupting it should break the task.
        acts = {}
        for a in w.agents:
            o = np.asarray(obs[a], float).reshape(-1)
            if a.startswith("speaker"):
                acts[a] = int(np.argmax(o)) if o.size else 0           # say the goal
            else:  # listener: decode goal, move to that landmark
                comm = o[-3:] if o.size >= 3 else np.zeros(3)
                if np.allclose(comm, 0.0):
                    acts[a] = 0                                         # no signal -> wait
                else:
                    g = int(np.argmax(comm))
                    rel = o[2 + 2 * g: 2 + 2 * g + 2]                   # rel pos of landmark g
                    acts[a] = _move_toward(rel) if rel.size == 2 else 0
        return acts

    def scripted_reference(w, obs):
        # Symmetric: each agent both speaks (says its partner's goal) and moves to
        # its own goal landmark using the received message.
        acts = {}
        for a in w.agents:
            o = np.asarray(obs[a], float).reshape(-1)
            comm = o[-3:] if o.size >= 3 else np.zeros(3)
            if np.allclose(comm, 0.0):
                acts[a] = 0
            else:
                g = int(np.argmax(comm))
                rel = o[2 + 2 * g: 2 + 2 * g + 2]
                acts[a] = _move_toward(rel) if rel.size == 2 else 0
        return acts

    def scripted_spread(w, obs):
        # Greedy coverage: move to nearest landmark (communication not essential).
        acts = {}
        for a in w.agents:
            o = np.asarray(obs[a], float).reshape(-1)
            # spread obs: self_vel(2), self_pos(2), landmark_rel(2*N), other_rel...
            lm = o[4:4 + 2 * 3].reshape(-1, 2) if o.size >= 10 else np.zeros((1, 2))
            d = np.linalg.norm(lm, axis=1)
            rel = lm[int(np.argmin(d))]
            acts[a] = _move_toward(rel)
        return acts

    def randompi(w, obs):
        return {a: w.action_space(a).sample() for a in w.agents}

    scripted_map = {
        "speaker_listener": scripted_speaker_listener,
        "simple_reference": scripted_reference,
        "simple_spread": scripted_spread,
    }
    policy = scripted_map.get(env_name, randompi) if algo == "scripted" else randompi
    rec = _run_wrapped(env, adapter, cfg, seed, policy, max_steps=max_cycles)
    rec["benchmark"] = f"mpe_{env_name}"; rec["algo"] = algo
    rec["n_agents"] = len(env.possible_agents)
    return rec


# ============================================================
# Task registry
# ============================================================
# Each entry: name -> dict(run=callable, defaults=dict of env kwargs/algo)
TASKS: Dict[str, Dict[str, Any]] = {
    # consensus family
    "consensus_mean":    dict(run=run_consensus, defaults=dict(algo="mean", n=20, d=1, steps=60)),
    "consensus_median":  dict(run=run_consensus, defaults=dict(algo="median", n=20, d=1, steps=60)),
    "consensus_trimmed": dict(run=run_consensus, defaults=dict(algo="trimmed", n=20, d=1, steps=60)),
    "rendezvous_mean":   dict(run=run_rendezvous, defaults=dict(algo="mean", n=20, steps=60)),
    "rendezvous_median": dict(run=run_rendezvous, defaults=dict(algo="median", n=20, steps=60)),
    "formation_mean":    dict(run=run_formation, defaults=dict(algo="mean", n=12, steps=80)),
    "formation_median":  dict(run=run_formation, defaults=dict(algo="median", n=12, steps=80)),
    "flocking_mean":     dict(run=run_flocking, defaults=dict(algo="mean", n=40, steps=60)),
    # dcop family
    "dcop_minsum":       dict(run=run_dcop, defaults=dict(algo="minsum", n=24, k_colors=4, steps=40, p_extra=0.08)),
    "dcop_dsa":          dict(run=run_dcop, defaults=dict(algo="dsa", n=24, k_colors=4, steps=40, p_extra=0.08)),
    # real benchmarks
    "wntr_rule":         dict(run=run_wntr, defaults=dict(algo="rule_based", horizon=24)),
    "wntr_random":       dict(run=run_wntr, defaults=dict(algo="random", horizon=24)),
    "mpe_speaker_listener": dict(run=run_mpe, defaults=dict(env_name="speaker_listener", algo="scripted", max_cycles=25)),
    "mpe_simple_reference": dict(run=run_mpe, defaults=dict(env_name="simple_reference", algo="scripted", max_cycles=25)),
    "mpe_simple_spread":    dict(run=run_mpe, defaults=dict(env_name="simple_spread", algo="scripted", max_cycles=25)),
}


def run_task(task_name: str, cfg: NetworkFaultConfig, seed: int, env_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = TASKS[task_name]
    kw = dict(spec["defaults"])
    if env_overrides:
        kw.update(env_overrides)
    rec = spec["run"](cfg, seed, **kw)
    rec.setdefault("task", task_name)
    rec["task_name"] = task_name
    return rec

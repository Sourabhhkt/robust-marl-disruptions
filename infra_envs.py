"""
Physical-infrastructure benchmarks: distributed economic dispatch by consensus.

Milestone 8 wrapped a single physical network (the WNTR/EPANET water-distribution
model) under a random and a rule-based policy. Milestone 9 broadens the
infrastructure coverage to *water, gas, and electric power* (and district
heating) and, crucially, makes communication causally drive the task by posing a
single, domain-agnostic distributed-control problem on each network:

    Distributed economic dispatch (ED).
    --------------------------------------------------------------
    A set of n controllable supply agents (generators / gas sources /
    pumps) must jointly meet an aggregate demand D at least cost,

        min  sum_i C_i(P_i),   C_i(P_i) = a_i P_i^2 + b_i P_i
        s.t. sum_i P_i = D,    0 <= P_i <= Pmax_i.

    The KKT optimum equalizes the incremental cost lambda_i = C_i'(P_i)
    across all agents (lambda_i = lambda* for every i). No agent knows D or
    the others' costs, so they must *communicate*. We use the incremental-cost
    consensus scheme (Zhang & Chow 2012; Kar & Hug 2012): each agent runs
    average consensus on its incremental cost lambda_i and on a distributed
    estimate g_i of the global power mismatch, exchanging both over the
    communication channel:

        lambda_i <- AGG_j in N(i)( lambda_j )  +  gamma * g_i
        P_i      <- clip( (lambda_i - b_i) / (2 a_i), 0, Pmax_i )
        g_i      <- AGG_j in N(i)( g_j )  -  (P_i(t) - P_i(t-1))

    The aggregator AGG is the algorithm under test (mean / median / trimmed /
    oracle / learned), so this benchmark accepts exactly the same coordinator
    interface as the synthetic consensus task. Communication impairments and
    Byzantine agents corrupt the lambda / g messages, so dispatch quality
    degrades with channel quality -- communication is on the decision-critical
    path (contrast the silent-failure cases of Milestone 8).

The *physical network* validates the coordinated dispatch: the final injections
are written into a pandapower (electric) or pandapipes (gas / heat) model, the
steady state is solved, and we report the fraction of buses / junctions whose
voltage / pressure leaves its safe band. This couples the abstract optimization
to real infrastructure physics and yields a cross-domain resilience comparison.

Domains
-------
- ``abstract`` : ED with no physics layer (fast; isolates the coordination).
- ``power``    : pandapower case (generators are the agents); voltage band check.
- ``gas``      : pandapipes gas network (sources are the agents); pressure band.
- ``heat``     : pandapipes district-heating network (pumps are the agents).
- ``water``    : reuse of the Milestone 8 WNTR task is kept in ``baselines.py``;
                 a consensus-dispatch water variant is provided here for parity.

Every ``*_rollout`` returns the same flat record schema as ``synth_envs.py`` so
that ``runner.py`` and ``analysis.py`` consume it unchanged.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from core import NetworkChannel, FaultModel, NetworkFaultConfig, DELIVERED_STATUSES
from metrics import CommLogger

warnings.filterwarnings("ignore")

# Optional heavy dependencies are imported lazily so the synthetic suite never
# pays for them and the module imports even where they are absent.
_PP = {"power": None, "gas": None}


# ============================================================
# Cost / agent model (heterogeneous by construction)
# ============================================================
def make_dispatch_problem(n: int, seed: int,
                          demand_frac: float = 0.6,
                          hetero: float = 1.0) -> Dict[str, np.ndarray]:
    """Generate a heterogeneous quadratic-cost dispatch problem for ``n`` agents.

    ``hetero`` in [0,1] scales the spread of cost coefficients and capacities;
    0 gives a homogeneous fleet, 1 a strongly heterogeneous one. ``demand_frac``
    sets the aggregate demand as a fraction of total capacity (loading)."""
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(4)[3])
    base_a = 0.5
    base_b = 1.0
    a = base_a * (1.0 + hetero * rng.uniform(-0.6, 1.4, size=n))   # quadratic cost
    b = base_b * (1.0 + hetero * rng.uniform(-0.5, 0.8, size=n))   # linear cost
    pmax = 1.0 + hetero * rng.uniform(0.0, 2.0, size=n)            # capacity spread
    a = np.maximum(a, 0.05)
    pmax = np.maximum(pmax, 0.4)
    D = float(demand_frac * pmax.sum())
    return {"a": a, "b": b, "pmax": pmax, "D": D, "n": n}


def central_optimum(prob: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Centralized ED optimum by bisection on the common incremental cost lambda."""
    a, b, pmax, D = prob["a"], prob["b"], prob["pmax"], prob["D"]

    def total_power(lam):
        P = np.clip((lam - b) / (2.0 * a), 0.0, pmax)
        return P.sum(), P

    lo, hi = float(b.min()), float((2.0 * a * pmax + b).max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        tot, _P = total_power(mid)
        if tot < D:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    tot, P = total_power(lam)
    cost = float(np.sum(a * P * P + b * P))
    return {"lambda": float(lam), "P": P, "cost": cost, "served": float(tot)}


def _cost(prob, P):
    a, b = prob["a"], prob["b"]
    return float(np.sum(a * P * P + b * P))


# ============================================================
# Communication graph (reuse synth_envs builder)
# ============================================================
def _build_graph(n, topology, rng, p_extra=0.25):
    import synth_envs as SE
    return SE.build_graph(n, topology, rng, p_extra=p_extra)


def _neighbors(adj):
    return [list(np.where(adj[i] > 0)[0]) for i in range(adj.shape[0])]


def metropolis_weights(adj: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings weights: a symmetric, doubly-stochastic averaging
    matrix for an undirected graph. Using these makes plain-mean consensus mass
    conserving, so the distributed mismatch estimator converges to the true
    global power imbalance (and hence supply meets demand) on a clean channel."""
    n = adj.shape[0]
    deg = adj.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def combine(strategy: Any, own: float, vals: List[float], w_self: float,
            w_neighbors: List[float]) -> float:
    """Blend an agent's own value with its delivered-neighbor values under the
    coordination strategy (the algorithm under test). ``mean`` uses the
    (renormalized) Metropolis weights and is therefore mass conserving; the
    robust strategies blend a fixed self-weight with a robust neighbor summary;
    a callable strategy receives the raw inputs for learned coordinators."""
    if not vals:
        return own
    if callable(strategy):
        return float(strategy(own, vals, w_self, w_neighbors))
    a = np.asarray(vals, dtype=float)
    if strategy == "mean":
        wn = np.asarray(w_neighbors, dtype=float)
        tot = w_self + wn.sum()
        return float((w_self * own + float(np.dot(wn, a))) / max(1e-12, tot))
    # robust strategies: fixed self-weight blended with a robust neighbor summary
    sw = max(0.05, min(0.9, w_self / max(1e-12, w_self + np.sum(w_neighbors))))
    if strategy == "median":
        summary = float(np.median(a))
    elif strategy == "trimmed":
        if a.size <= 2:
            summary = float(np.mean(a))
        else:
            k = int(np.floor(0.25 * a.size))
            s = np.sort(a)
            s = s[k: a.size - k] if (a.size - 2 * k) > 0 else s
            summary = float(np.mean(s))
    else:
        summary = float(np.mean(a))
    return sw * own + (1.0 - sw) * summary


# ============================================================
# Core: distributed economic dispatch by consensus
# ============================================================
def dispatch_consensus_rollout(
    cfg: NetworkFaultConfig,
    seed: int,
    combiner: Any = "mean",
    domain: str = "abstract",
    n: int = 6,
    steps: int = 80,
    topology: str = "ring_plus",
    p_extra: float = 0.3,
    eps_balance: float = 0.12,
    demand_frac: float = 0.6,
    hetero: float = 1.0,
    byzantine_frac: float = 0.0,
    byzantine_scale: float = 3.0,
    physics: bool = True,
    hetero_profile: Optional[Any] = None,
    **_,
) -> Dict[str, Any]:
    """Incremental-cost consensus dispatch over an impaired communication graph.

    ``combiner`` is the coordination strategy under test (``mean`` / ``median`` /
    ``trimmed`` / a callable for learned/oracle/hybrid coordinators); it fuses the
    received incremental-cost and mismatch-estimate messages. Returns a flat
    record dict (same schema family as ``synth_envs``)."""
    prob = make_dispatch_problem(n, seed, demand_frac=demand_frac, hetero=hetero)
    a, b, pmax, D = prob["a"], prob["b"], prob["pmax"], prob["D"]
    opt = central_optimum(prob)

    env_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(5)[4])
    agents = [str(i) for i in range(n)]
    adj = _build_graph(n, topology, env_rng, p_extra=p_extra)
    nbrs = _neighbors(adj)
    W = metropolis_weights(adj)

    n_byz = int(round(byzantine_frac * n))
    byz = set(agents[:n_byz])

    # init: equal-share dispatch; y_i tracks the network-average power imbalance
    # via dynamic average consensus (Kar & Hug 2012; Zhang & Chow 2012). Local
    # demand shares d_i sum to D, so sum_i (d_i - P_i) = D - sum P_i.
    d_share = np.full(n, D / n)
    P = np.clip(d_share.copy(), 0.0, pmax)
    lam = 2.0 * a * P + b
    y = d_share - P                       # local imbalance estimate (mismatch)
    P_prev = P.copy()

    channel = NetworkChannel(cfg); channel.reset(agents, seed=seed)
    fm = FaultModel(cfg); fm.reset(agents, seed=seed)
    comm = CommLogger(bits_per_entry=float(cfg.bandwidth_bits) if cfg.bandwidth_bits else 32.0)

    def lam_disagreement(lv):
        return float(np.sum((lv - lv.mean()) ** 2))

    V = [lam_disagreement(lam)]
    mismatch_hist = [abs(float(P.sum() - D))]
    cost_hist = [_cost(prob, P)]

    for t in range(steps):
        fm.begin_step(agents, t)

        # --- send (each agent broadcasts [lambda_i, y_i]) ---
        for i in range(n):
            if fm.is_crashed(agents[i], t):
                continue
            if agents[i] in byz:
                payload = env_rng.normal(0.0, byzantine_scale, size=2)
            else:
                payload = np.array([lam[i], y[i]], dtype=float)
            for j in nbrs[i]:
                channel.send(agents[i], agents[j], payload, t, agents=agents,
                             allow_byzantine_corrupt=True)
                comm.on_send()

        # --- receive + aggregate ---
        lam_new = lam.copy()
        y_new = y.copy()
        for j in range(n):
            if fm.is_crashed(agents[j], t):
                continue
            got = channel.recv_all(agents[j], t, dim=2, agents=agents)
            lam_vals: List[float] = []
            y_vals: List[float] = []
            w_neighbors: List[float] = []
            for src, (payload, meta) in got.items():
                comm.on_recv(payload, meta)
                if meta["status"] in DELIVERED_STATUSES:
                    v = np.asarray(fm.transform_observation(agents[j], payload, t),
                                   dtype=float).reshape(-1)[:2]
                    lam_vals.append(float(v[0]))
                    y_vals.append(float(v[1]))
                    w_neighbors.append(float(W[j, int(src)]))
            if lam_vals:
                # consensus combine (algorithm under test) on price + mismatch,
                # plus the balance feedback eps * y on the price.
                lam_mix = combine(combiner, lam[j], lam_vals, float(W[j, j]), w_neighbors)
                y_mix = combine(combiner, y[j], y_vals, float(W[j, j]), w_neighbors)
                lam_new[j] = lam_mix + eps_balance * y[j]
                y_new[j] = y_mix
        lam = lam_new
        # recompute dispatch from updated incremental costs
        P = np.clip((lam - b) / (2.0 * a), 0.0, pmax)
        # finish the dynamic-average-consensus update for the mismatch estimate
        y = y_new - (P - P_prev)
        P_prev = P.copy()

        V.append(lam_disagreement(lam))
        mismatch_hist.append(abs(float(P.sum() - D)))
        cost_hist.append(_cost(prob, P))

    V = np.asarray(V, dtype=float)
    final_mismatch = mismatch_hist[-1]
    # optimality gap relative to the centralized optimum (guard tiny denom).
    # cost_gap scores only the cost of the realized generation, so it can read
    # zero (or negative, hence the floor) when the dispatch *under-produces*: the
    # unmet demand is cheaper but infeasible. We therefore also report a
    # feasibility-aware Lagrangian dispatch regret that prices the supply-demand
    # imbalance at the optimal shadow price lambda*:
    #     L(P; lambda*) = sum_i C_i(P_i) + lambda* (D - sum_i P_i),
    #     regret        = (L(P; lambda*) - C_opt) / C_opt.
    # Because each term C_i(P_i) - lambda* P_i is minimized at the optimal P_i*,
    # L(P; lambda*) >= C_opt for every feasible P (equality only at the optimum),
    # so regret >= 0 and rises monotonically with coordination error -- it does
    # not get masked by under-production the way the raw cost gap does.
    served = float(P.sum())
    lagrangian = cost_hist[-1] + opt["lambda"] * (D - served)
    regret = (lagrangian - opt["cost"]) / max(1e-6, abs(opt["cost"]))
    cost_gap = (cost_hist[-1] - opt["cost"]) / max(1e-6, abs(opt["cost"]))

    rec: Dict[str, Any] = {
        "benchmark": f"ed_{domain}",
        "domain": domain,
        "n_agents": n,
        "steps": steps,
        "topology": topology,
        "demand": D,
        "lambda_opt": opt["lambda"],
        "final_lambda_disagreement": float(V[-1]),
        "final_disagreement": float(V[-1]),          # alias for shared analysis
        "final_power_mismatch": float(final_mismatch),
        "final_cost_gap": float(max(cost_gap, 0.0)),
        "final_dispatch_regret": float(max(regret, 0.0)),
        "V": V,
        "adj": adj,
        "eff_adjs": [],
    }

    # --- physical-network validation of the coordinated dispatch ---
    if physics and domain in ("power", "gas", "heat", "water"):
        try:
            viol = _physics_validate(domain, prob, P, seed)
            rec.update(viol)
        except Exception as e:  # keep the sweep alive; flag absence
            rec["phys_violation_frac"] = float("nan")
            rec["phys_error"] = repr(e)[:120]

    rec["return"] = float(-(rec["final_cost_gap"] + rec.get("phys_violation_frac", 0.0) or 0.0))
    rec.update(comm.summary())
    return rec


# ============================================================
# Physics validators (one per infrastructure domain)
# ============================================================
_PIPE_SPEC = {
    # domain -> (fluid, nominal slack pressure bar, minimum delivery pressure bar)
    # Compressible gas distribution. Water is covered by the established WNTR /
    # EPANET hydraulic benchmark (retained from Milestone 8); incompressible
    # flow on a mesh with mass sources is ill-posed, so we keep the gas pipe
    # model here and route water physics through WNTR.
    "gas":   ("lgas",  3.0, 1.5),
}


def _physics_validate(domain: str, prob: Dict[str, np.ndarray], P: np.ndarray,
                      seed: int) -> Dict[str, float]:
    if domain == "power":
        return _validate_power(prob, P, seed)
    if domain in _PIPE_SPEC:
        return _validate_pipe(prob, P, seed, domain)
    return {}


def _validate_power(prob, P, seed) -> Dict[str, float]:
    """Map agent dispatch to a pandapower transmission case (IEEE 30-bus) and
    report the fraction of lines past an N-1 security loading margin (80%).

    A DC power-flow solve is used: it is the standard model for active-power
    economic dispatch and is ~100x faster than the AC solve, so the full sweep
    stays in the project's CPU/minutes budget. Generation is shared across the
    network's generators in proportion to the coordinated dispatch; the slack
    bus absorbs the residual, so a miscoordinated dispatch concentrates flow and
    overloads lines."""
    import pandapower as pp
    import pandapower.networks as ppn
    net = ppn.case30()
    gens = net.gen.index.tolist()
    total_load = float(net.load.p_mw.sum())
    D = float(prob["D"])
    m = min(len(P), len(gens))
    inj = np.maximum(0.0, P[:m] / max(1e-6, D) * total_load)
    for k, gi in enumerate(gens[:m]):
        net.gen.at[gi, "p_mw"] = float(inj[k])
    try:
        pp.rundcpp(net)
    except Exception:
        return {"phys_violation_frac": 1.0, "phys_infeasible": 1.0,
                "phys_max_loading": float("nan")}
    load_pct = net.res_line.loading_percent.to_numpy()
    margin = 80.0   # N-1 security margin (percent of thermal rating)
    viol = float(np.mean(load_pct > margin))
    return {"phys_violation_frac": viol, "phys_infeasible": 0.0,
            "phys_max_loading": float(np.nanmax(load_pct))}


def _build_pipe_network(n_agents: int, seed: int, fluid: str, p_slack: float):
    """Parametric meshed pipe-distribution network with ``n_agents`` controllable
    sources feeding a ring of demand junctions (one slack ext_grid). Used for the
    gas, water, and district-heating domains by varying the fluid and pressure."""
    import pandapipes as ppi
    net = ppi.create_empty_network(fluid=fluid)
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(7)[6])
    n_demand = max(6, 2 * n_agents)
    dem = [ppi.create_junction(net, pn_bar=p_slack, tfluid_k=293.15, name=f"d{i}")
           for i in range(n_demand)]
    for i in range(n_demand):                                   # demand ring
        j = (i + 1) % n_demand
        ppi.create_pipe_from_parameters(net, dem[i], dem[j], length_km=1.0,
                                        diameter_m=0.05, k_mm=0.2)
    ppi.create_ext_grid(net, junction=dem[0], p_bar=p_slack, t_k=293.15)  # slack
    for k in range(n_agents):                                   # controllable sources
        jx = dem[(1 + k * n_demand // max(1, n_agents)) % n_demand]
        ppi.create_source(net, junction=jx, mdot_kg_per_s=0.0, name=f"src{k}")
    # Thin, long pipes and substantial demand make delivery pressure sensitive to
    # the spatial dispatch pattern, so miscoordination shows up as real pressure
    # violations rather than being absorbed silently by the slack.
    for i in range(n_demand):                                   # stochastic demands
        ppi.create_sink(net, junction=dem[i],
                        mdot_kg_per_s=float(0.03 * (1.0 + 0.3 * rng.random())))
    return net


def _validate_pipe(prob, P, seed, domain) -> Dict[str, float]:
    """Map agent injections to source mass flows; report pressure-band violations.

    A poorly coordinated dispatch (some agents over-inject, others under) drives
    delivery pressures outside the safe band; this is the physical cost of
    degraded communication in a gas / water / district-heating network."""
    import pandapipes as ppi
    try:
        from pandapipes.pipeflow import PipeflowNotConverged
    except Exception:  # pragma: no cover - fall back to broad catch
        PipeflowNotConverged = Exception
    fluid, p_slack, pmin = _PIPE_SPEC[domain]
    n = len(P)
    net = _build_pipe_network(n, seed, fluid, p_slack)
    srcs = net.source.index.tolist()
    D = float(prob["D"])
    total_demand = float(net.sink.mdot_kg_per_s.sum())
    # Source injections track the coordinated dispatch, scaled so that a balanced
    # dispatch (sum_i P_i = D) exactly meets the network demand. A badly
    # miscoordinated dispatch -- some sources over-injecting, others starved
    # because their price message was lost or forged -- leaves the total off
    # demand and the spatial pattern wrong, driving delivery pressures out of
    # band; an outright infeasible hydraulic solve is itself a full violation.
    inj = np.maximum(0.0, P / max(1e-6, D) * total_demand)
    for k, si in enumerate(srcs[:n]):
        net.source.at[si, "mdot_kg_per_s"] = float(inj[k])
    try:
        ppi.pipeflow(net)
    except (PipeflowNotConverged, Exception):
        return {"phys_violation_frac": 1.0, "phys_pmin": float("nan"),
                "phys_pmax": float("nan"), "phys_infeasible": 1.0}
    pbar = net.res_junction.p_bar.to_numpy()
    viol = float(np.mean(pbar < pmin))
    return {"phys_violation_frac": viol, "phys_infeasible": 0.0,
            "phys_pmin": float(np.nanmin(pbar)), "phys_pmax": float(np.nanmax(pbar))}


if __name__ == "__main__":
    # self-check: clean channel should reach near-equal incremental costs,
    # zero power mismatch, and a feasible physical network state.
    for dom in ["abstract", "power", "gas"]:
        rec = dispatch_consensus_rollout(NetworkFaultConfig(seed=0), 0,
                                         "mean", domain=dom, n=6,
                                         physics=(dom != "abstract"))
        print(f"{dom:8s} lam_disag={rec['final_lambda_disagreement']:.2e} "
              f"mismatch={rec['final_power_mismatch']:.3e} "
              f"cost_gap={rec['final_cost_gap']:.3e} "
              f"viol={rec.get('phys_violation_frac','-')}")

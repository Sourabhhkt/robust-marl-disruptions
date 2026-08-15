"""
Coordinated consensus rollout for Milestone 9.

A superset of ``synth_envs.linear_consensus_rollout`` that drives the seven
coordination strategies of Milestone 9 (classical mean / median / trimmed,
oracle, graph-attention GNN, reward-redistribution RECO, and hybrid MAS+DPS) over
heterogeneous teams and the same channel/fault engine. The Milestone 8 rollout is
left untouched so its results reproduce exactly; this module is the new code path
for the comparative study.

Record schema matches ``synth_envs`` so ``runner.py`` / ``analysis.py`` consume it
unchanged. Additional fields: ``msgs_sent`` (communication volume, for the
event-triggered / hybrid efficiency comparison) and ``strategy``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from core import (NetworkChannel, FaultModel, NetworkFaultConfig,
                  DELIVERED_STATUSES, quantize_uniform)
from metrics import CommLogger, disagreement
import synth_envs as SE
import baselines as B
import coordinators as C
from hetero import HeteroProfile, homogeneous_profile


def _resolve_aggregator(strategy: Any):
    """Map a strategy to an ``aggregator(own, vals)`` callable, or a marker tag
    ('oracle' / 'hybrid') handled structurally by the rollout."""
    if isinstance(strategy, str):
        if strategy in B.AGGREGATORS:
            return B.AGGREGATORS[strategy], None
        if strategy in ("oracle", "hybrid"):
            return None, strategy
        raise KeyError(f"unknown strategy {strategy}")
    if isinstance(strategy, C.OracleCoordinator):
        return None, "oracle"
    # learned aggregator or any callable(own, vals)
    return strategy, None


def coordinated_consensus_rollout(
    cfg: NetworkFaultConfig,
    seed: int,
    strategy: Any = "mean",
    n: int = 20,
    d: int = 1,
    steps: int = 60,
    topology: str = "ring_plus",
    p_extra: float = 0.15,
    step_size: float = 0.35,
    init_scale: float = 1.0,
    init_offset: float = 0.0,
    offsets: Optional[np.ndarray] = None,
    byzantine_frac: float = 0.0,
    byzantine_scale: float = 3.0,
    byz_attack: str = "gauss",
    task: str = "consensus",
    hetero_profile: Optional[HeteroProfile] = None,
    n_clusters: Optional[int] = None,
    event_trigger: float = 0.0,
    **_,
) -> Dict[str, Any]:
    env_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(3)[2])
    # Heterogeneity (per-agent transmission reliability) draws from a *dedicated*
    # independent sub-stream, so the graph / initial-state / Byzantine-payload
    # draws taken from env_rng are byte-for-byte identical to the Milestone 8
    # rollout. This keeps the homogeneous baselines reproducible against M8 and,
    # for any fixed seed, makes the whole suite deterministically reproducible.
    het_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(11)[10])
    agents = [str(i) for i in range(n)]
    adj = SE.build_graph(n, topology, env_rng, p_extra=p_extra)

    aggregator, marker = _resolve_aggregator(strategy)
    prof = hetero_profile if hetero_profile is not None else homogeneous_profile(n)

    # --- hybrid MAS+DPS: cluster the graph, restrict edges to intra-cluster +
    #     leader-to-leader, and flag leaders for inter-cluster carry. ---
    cluster = None
    if marker == "hybrid":
        k = n_clusters or max(2, int(round(np.sqrt(n))))
        cluster, leaders = C.hybrid_clusters(adj, k)
        leader_of = {c: leaders[c] for c in range(len(leaders))}
        hadj = np.zeros_like(adj)
        for i in range(n):
            # star within cluster: every member links to its cluster leader, so
            # the intra-cluster diameter is 2 regardless of cluster size.
            li = leader_of[int(cluster[i])]
            if li != i:
                hadj[i, li] = hadj[li, i] = 1.0
        for li in leaders:                      # leaders form a clique (fast backbone)
            for lj in leaders:
                if li != lj:
                    hadj[li, lj] = 1.0
        adj = hadj
        prof = HeteroProfile(prof.gain_scale, prof.reliability, prof.precision_bits,
                             np.isin(np.arange(n), leaders))
        aggregator = B.AGGREGATORS["mean"]      # hybrid uses mean within structure

    nbrs = [list(np.where(adj[i] > 0)[0]) for i in range(n)]

    n_byz = int(round(byzantine_frac * n))
    byz = set(agents[:n_byz])
    honest_mask = np.array([a not in byz for a in agents])

    X = env_rng.normal(0.0, init_scale, size=(n, d)) + float(init_offset)
    if offsets is not None:
        X = X + offsets
    # Ideal unquantized consensus target = mean of the honest agents' initial
    # states (the value a clean, full-precision channel would converge to). Used
    # for the rate-distortion consensus-error metric, matching Milestone 8: unlike
    # the inter-agent disagreement, this exposes a common-but-biased quantization
    # lattice lock (agents agreeing exactly on the wrong value).
    true_target = X[honest_mask].mean(axis=0) if honest_mask.any() else X.mean(axis=0)

    channel = NetworkChannel(cfg); channel.reset(agents, seed=seed)
    fm = FaultModel(cfg); fm.reset(agents, seed=seed)
    bits = float(cfg.bandwidth_bits) if cfg.bandwidth_bits else 32.0
    comm = CommLogger(bits_per_entry=bits)

    def lyap(Xt):
        if offsets is not None:
            return SE._formation_error(Xt, adj, offsets)
        return float(disagreement(Xt[None])[0])

    # oracle link state (delayed honest-mean delivery under the channel budget)
    oracle_hist: List[np.ndarray] = []
    # event-triggered messaging: init to +inf so the first transmission always
    # fires, then suppress sends until the state moves by more than the threshold.
    last_sent = np.full_like(X, np.inf)

    V = [lyap(X)]
    eff_adjs: List[np.ndarray] = []
    msgs_sent = 0

    for t in range(steps):
        fm.begin_step(agents, t)

        # ---------- oracle coordinator ----------
        if marker == "oracle":
            target = C.oracle_consensus_target(X, honest_mask)     # honest global mean
            oracle_hist.append(target.copy())
            lat = int(getattr(cfg, "base_latency_steps", 0) or 0)
            src_val = oracle_hist[max(0, len(oracle_hist) - 1 - lat)]
            if cfg.bandwidth_bits:
                src_val = quantize_uniform(src_val, int(cfg.bandwidth_bits), cfg.quant_clip)
            newX = X.copy()
            for j in range(n):
                if fm.is_crashed(agents[j], t):
                    continue
                comm.on_send(); msgs_sent += 1
                # oracle link still suffers loss; on a lost step the agent holds
                if env_rng.random() < float(cfg.msg_drop_prob):
                    continue
                g = np.asarray(src_val, dtype=float).reshape(-1)[:d]
                newX[j] = X[j] + step_size * float(prof.gain_scale[j]) * (g - X[j])
            X = newX
            V.append(lyap(X)); eff_adjs.append(np.zeros((n, n)))
            continue

        # ---------- send (classical / learned / hybrid) ----------
        for i in range(n):
            if fm.is_crashed(agents[i], t):
                continue
            # heterogeneous link reliability: a degraded radio may skip this step
            if het_rng.random() > float(prof.reliability[i]):
                continue
            # event-triggered messaging: suppress if state barely changed
            if event_trigger > 0.0 and np.linalg.norm(X[i] - last_sent[i]) < event_trigger:
                continue
            last_sent[i] = X[i].copy()
            if agents[i] in byz:
                # Byzantine attack signal. 'gauss' (the training distribution for
                # the learned aggregators) is wide zero-mean noise; 'bias' and
                # 'signflip' are off-distribution attacks used to test whether the
                # learned robustness transfers to unseen adversaries.
                if byz_attack == "bias":
                    payload = X[i].copy() + byzantine_scale          # stealthy constant offset
                elif byz_attack == "signflip":
                    payload = -byzantine_scale * X[i].copy()         # negated, amplified
                else:
                    payload = env_rng.normal(0.0, byzantine_scale, size=d)
            else:
                payload = X[i].copy()
                # heterogeneous per-agent precision (bandwidth-starved nodes)
                pb = int(prof.precision_bits[i])
                if pb > 0:
                    payload = quantize_uniform(payload, pb, cfg.quant_clip)
            for j in nbrs[i]:
                channel.send(agents[i], agents[j], payload, t, agents=agents,
                             allow_byzantine_corrupt=True)
                comm.on_send(); msgs_sent += 1

        # ---------- receive + update ----------
        eff = np.zeros((n, n), dtype=float)
        newX = X.copy()
        for j in range(n):
            if fm.is_crashed(agents[j], t):
                continue
            got = channel.recv_all(agents[j], t, dim=d, agents=agents)
            targets: List[np.ndarray] = []
            for src, (payload, meta) in got.items():
                comm.on_recv(payload, meta)
                if meta["status"] in DELIVERED_STATUSES:
                    v = np.asarray(fm.transform_observation(agents[j], payload, t),
                                   dtype=float).reshape(-1)[:d]
                    if offsets is not None:
                        si = int(src)
                        v = v - (offsets[si] - offsets[j])
                    targets.append(v)
                    si = int(src)
                    eff[j, si] = 1.0; eff[si, j] = 1.0
            if targets:
                g = np.asarray(aggregator(X[j], targets), dtype=float).reshape(-1)[:d]
                newX[j] = X[j] + step_size * float(prof.gain_scale[j]) * (g - X[j])
        X = newX
        V.append(lyap(X)); eff_adjs.append(eff)

    V = np.asarray(V, dtype=float)
    consensus_error = float(np.mean(np.sum((X - true_target) ** 2, axis=1)))
    rec: Dict[str, Any] = {
        "benchmark": task,
        "strategy": getattr(strategy, "name", strategy if isinstance(strategy, str) else "learned"),
        "n_agents": n,
        "steps": steps,
        "topology": topology,
        "return": float(-V[-1]),
        "final_disagreement": float(V[-1]),
        "consensus_error": consensus_error,
        "msgs_sent": int(msgs_sent),
        "V": V,
        "adj": adj,
        "eff_adjs": eff_adjs,
        "nominal_lambda2": None,
    }
    rec.update(comm.summary())
    return rec

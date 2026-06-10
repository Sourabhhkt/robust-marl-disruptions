"""
Synthetic, communication-native MAS/DPS benchmarks.

Unlike the wrapped real benchmarks (PettingZoo MPE, WNTR), these environments
use the communication channel *directly*: each agent broadcasts an explicit
message to its graph neighbours and acts on what it receives. Messages
therefore causally drive the dynamics and the task outcome, so communication
impairments (drop, latency, quantization, spoofing, jamming, partition) and
agent faults (crash, sensor noise, Byzantine) produce real performance changes
rather than the silent distortion that can otherwise mask channel degradation.

All environments share the channel/fault engine in ``core.py`` and report
communication statistics via ``metrics.CommLogger``. Each ``*_rollout`` returns
a record dict consumed by ``runner.py`` and ``analysis.py``.

Families
--------
- consensus / rendezvous / formation / flocking : ``linear_consensus_rollout``
- distributed constraint optimization (graph colouring) : ``dcop_rollout``
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np

from core import NetworkChannel, FaultModel, NetworkFaultConfig, DELIVERED_STATUSES
from metrics import CommLogger, disagreement


# ============================================================
# Communication graph builders
# ============================================================
def build_graph(n: int, topology: str, rng: np.random.Generator, p_extra: float = 0.15) -> np.ndarray:
    """Return a symmetric 0/1 adjacency matrix (connected) for ``n`` agents."""
    A = np.zeros((n, n), dtype=float)
    if n <= 1:
        return A

    if topology == "complete":
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
        return A

    # start from a ring (guarantees connectivity)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = A[j, i] = 1.0

    if topology == "ring":
        return A

    if topology in ("ring_plus", "random"):
        # add random chords
        for i in range(n):
            for j in range(i + 2, n):
                if rng.random() < p_extra:
                    A[i, j] = A[j, i] = 1.0
        return A

    if topology == "star":
        A[:] = 0.0
        for i in range(1, n):
            A[0, i] = A[i, 0] = 1.0
        return A

    if topology == "line":
        A[:] = 0.0
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1.0
        return A

    return A  # fallback: ring


def _neighbors(adj: np.ndarray) -> List[List[int]]:
    return [list(np.where(adj[i] > 0)[0]) for i in range(adj.shape[0])]


def _formation_offsets(n: int, radius: float = 1.0) -> np.ndarray:
    """Regular-polygon target offsets in R^2 (used by the formation task)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)


def _formation_error(X: np.ndarray, adj: np.ndarray, offsets: np.ndarray) -> float:
    """Sum over edges of ||(x_i - x_j) - (o_i - o_j)||^2."""
    n = X.shape[0]
    err = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                d = (X[i] - X[j]) - (offsets[i] - offsets[j])
                err += float(np.dot(d, d))
    return err


# ============================================================
# Consensus / rendezvous / formation / flocking
# ============================================================
def linear_consensus_rollout(
    cfg: NetworkFaultConfig,
    seed: int,
    aggregator: Callable[[np.ndarray, List[np.ndarray]], np.ndarray],
    n: int = 20,
    d: int = 2,
    steps: int = 60,
    topology: str = "ring_plus",
    p_extra: float = 0.15,
    step_size: float = 0.35,
    init_scale: float = 1.0,
    init_offset: float = 0.0,
    offsets: Optional[np.ndarray] = None,
    byzantine_frac: float = 0.0,
    byzantine_scale: float = 3.0,
    task: str = "consensus",
    **_,
) -> Dict[str, Any]:
    """
    First-order distributed coordination over a communication graph.

    Update (per agent i that received messages):
        target_j  = x_j_hat                      (consensus / rendezvous)
                  = x_j_hat - (o_j - o_i)        (formation)
        g_i       = aggregator(x_i, {target_j})
        x_i      += step_size * (g_i - x_i)

    The ``aggregator`` is the algorithm under test (mean / median / trimmed-mean
    / ...). Crashed agents freeze and do not transmit. Byzantine agents transmit
    adversarial noise.
    """
    env_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(3)[2])
    agents = [str(i) for i in range(n)]
    adj = build_graph(n, topology, env_rng, p_extra=p_extra)
    nbrs = _neighbors(adj)

    n_byz = int(round(byzantine_frac * n))
    byz = set(agents[:n_byz])

    X = env_rng.normal(0.0, init_scale, size=(n, d)) + float(init_offset)
    if offsets is not None:
        X = X + offsets  # start scattered around the formation

    channel = NetworkChannel(cfg); channel.reset(agents, seed=seed)
    fm = FaultModel(cfg); fm.reset(agents, seed=seed)
    bits = float(cfg.bandwidth_bits) if cfg.bandwidth_bits else 32.0
    comm = CommLogger(bits_per_entry=bits)

    def lyap(Xt):
        if offsets is not None:
            return _formation_error(Xt, adj, offsets)
        return float(disagreement(Xt[None])[0])

    traj = [X.copy()]
    V = [lyap(X)]
    eff_adjs: List[np.ndarray] = []

    for t in range(steps):
        fm.begin_step(agents, t)

        # --- send ---
        for i in range(n):
            if fm.is_crashed(agents[i], t):
                continue
            if agents[i] in byz:
                payload = env_rng.normal(0.0, byzantine_scale, size=d)
            else:
                payload = X[i].copy()
            for j in nbrs[i]:
                channel.send(agents[i], agents[j], payload, t, agents=agents,
                             allow_byzantine_corrupt=True)
                comm.on_send()

        # --- receive + update ---
        eff = np.zeros((n, n), dtype=float)
        newX = X.copy()
        for j in range(n):
            if fm.is_crashed(agents[j], t):
                continue  # frozen state
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
                    eff[j, si] = 1.0
                    eff[si, j] = 1.0
            if targets:
                g = np.asarray(aggregator(X[j], targets), dtype=float).reshape(-1)[:d]
                newX[j] = X[j] + step_size * (g - X[j])
        X = newX
        traj.append(X.copy())
        V.append(lyap(X))
        eff_adjs.append(eff)

    V = np.asarray(V, dtype=float)
    # Rate-distortion distortion: squared deviation of the achieved agreed value
    # from the ideal (unquantized) consensus value, i.e. the mean of the initial
    # states. Unlike the disagreement V (which can be zero when agents lock onto a
    # common but biased quantization level), this measures the bias from the true
    # mean and is monotone in the communication budget.
    true_target = traj[0].mean(axis=0)               # (d,)
    consensus_error = float(np.mean(np.sum((X - true_target) ** 2, axis=1)))
    rec: Dict[str, Any] = {
        "benchmark": task,
        "n_agents": n,
        "steps": steps,
        "topology": topology,
        "return": float(-V[-1]),
        "final_disagreement": float(V[-1]),
        "consensus_error": consensus_error,
        "V": V,
        "adj": adj,
        "eff_adjs": eff_adjs,
        "nominal_lambda2": None,  # filled by analysis if needed
    }
    rec.update(comm.summary())
    return rec


# ============================================================
# Distributed constraint optimization (graph colouring)
# ============================================================
def dcop_rollout(
    cfg: NetworkFaultConfig,
    seed: int,
    algorithm: str = "minsum",      # "minsum" (max-sum/BP) or "dsa"
    n: int = 20,
    k_colors: int = 3,
    steps: int = 40,
    topology: str = "ring_plus",
    p_extra: float = 0.25,
    dsa_p: float = 0.6,
    byzantine_frac: float = 0.0,
    **_,
) -> Dict[str, Any]:
    """
    Distributed graph colouring as a DCOP: each node picks one of ``k_colors``;
    the cost is the number of monochromatic edges (constraint violations).

    Two solvers exchange genuine messages over the channel:
      - ``dsa``    : Distributed Stochastic Algorithm. Message = current colour
                     (one-hot, dim k). Each round a node (w.p. ``dsa_p``) switches
                     to the colour minimizing conflicts with received neighbours.
      - ``minsum`` : Loopy min-sum belief propagation (the "max-sum" family).
                     Message = a cost vector over the receiver's colours (dim k).

    Communication impairments corrupt/drop these messages, so solution quality
    and convergence degrade with channel quality.
    """
    env_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(3)[2])
    agents = [str(i) for i in range(n)]
    adj = build_graph(n, topology, env_rng, p_extra=p_extra)
    nbrs = _neighbors(adj)
    n_byz = int(round(byzantine_frac * n))
    byz = set(agents[:n_byz])

    channel = NetworkChannel(cfg); channel.reset(agents, seed=seed)
    fm = FaultModel(cfg); fm.reset(agents, seed=seed)
    bits = float(cfg.bandwidth_bits) if cfg.bandwidth_bits else 32.0
    comm = CommLogger(bits_per_entry=bits)

    def n_conflicts(colors):
        c = 0
        for i in range(n):
            for j in nbrs[i]:
                if j > i and colors[i] == colors[j]:
                    c += 1
        return c

    colors = env_rng.integers(0, k_colors, size=n)
    conflicts = [n_conflicts(colors)]
    # small per-node unary potentials break the symmetry that would otherwise
    # leave loopy min-sum stuck at the all-zero (all-same-colour) fixed point.
    theta = env_rng.uniform(0.0, 0.1, size=(n, k_colors))
    # incoming-message memory for min-sum (m_{src->j} over j's colours)
    inbox_prev: Dict[int, Dict[int, np.ndarray]] = {j: {} for j in range(n)}

    for t in range(steps):
        fm.begin_step(agents, t)

        # --- send ---
        for i in range(n):
            if fm.is_crashed(agents[i], t):
                continue
            if algorithm == "dsa":
                msg = np.zeros(k_colors); msg[int(colors[i])] = 1.0
                if agents[i] in byz:
                    msg = np.zeros(k_colors); msg[int(env_rng.integers(0, k_colors))] = 1.0
                for j in nbrs[i]:
                    channel.send(agents[i], agents[j], msg, t, agents=agents,
                                 allow_byzantine_corrupt=True)
                    comm.on_send()
            else:  # minsum: message to j = min over c_i of (unary + incoming-from-others + pairwise)
                for j in nbrs[i]:
                    incoming = theta[i].copy()
                    for l in nbrs[i]:
                        if l != j and l in inbox_prev[i]:
                            incoming += inbox_prev[i][l]
                    # m_{i->j}(c_j) = min_{c_i} [ incoming(c_i) + penalty*(c_i==c_j) ]
                    m = np.empty(k_colors)
                    for cj in range(k_colors):
                        costs = incoming.copy()
                        costs[cj] += 1.0  # monochromatic penalty
                        m[cj] = costs.min()
                    m = m - m.mean()  # normalize
                    if agents[i] in byz:
                        m = env_rng.normal(0.0, 1.0, size=k_colors)
                    channel.send(agents[i], agents[j], m, t, agents=agents,
                                 allow_byzantine_corrupt=True)
                    comm.on_send()

        # --- receive ---
        received: Dict[int, Dict[int, np.ndarray]] = {j: {} for j in range(n)}
        for j in range(n):
            if fm.is_crashed(agents[j], t):
                continue
            got = channel.recv_all(agents[j], t, dim=k_colors, agents=agents)
            for src, (payload, meta) in got.items():
                comm.on_recv(payload, meta)
                if meta["status"] in DELIVERED_STATUSES:
                    received[j][int(src)] = np.asarray(payload, dtype=float).reshape(-1)[:k_colors]

        # --- update ---
        if algorithm == "dsa":
            newc = colors.copy()
            for j in range(n):
                if fm.is_crashed(agents[j], t):
                    continue
                if env_rng.random() >= dsa_p:
                    continue
                counts = np.zeros(k_colors)
                for src, vec in received[j].items():
                    c = int(np.argmax(vec))
                    counts[c] += 1.0
                if counts.sum() > 0:
                    newc[j] = int(np.argmin(counts))
            colors = newc
        else:  # minsum decision from accumulated beliefs
            inbox_prev = received
            for j in range(n):
                if fm.is_crashed(agents[j], t):
                    continue
                belief = theta[j].copy()
                for src, vec in received[j].items():
                    belief += vec
                if received[j]:
                    colors[j] = int(np.argmin(belief))

        conflicts.append(n_conflicts(colors))

    conflicts = np.asarray(conflicts, dtype=float)
    n_edges = int(adj.sum() / 2)
    rec: Dict[str, Any] = {
        "benchmark": f"dcop_{algorithm}",
        "n_agents": n,
        "steps": steps,
        "topology": topology,
        "n_edges": n_edges,
        "k_colors": k_colors,
        "return": float(-conflicts[-1]),
        "final_conflicts": float(conflicts[-1]),
        "final_conflict_frac": float(conflicts[-1] / max(1, n_edges)),
        "V": conflicts,                 # treat conflicts as the "energy" trajectory
        "final_disagreement": float(conflicts[-1]),
        "adj": adj,
        "eff_adjs": [],
    }
    rec.update(comm.summary())
    return rec

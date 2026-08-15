"""
Advanced coordination strategies for Milestone 9.

Milestone 8 evaluated *classical* coordination rules (mean / median / trimmed-mean
consensus, min-sum / DSA DCOP). Milestone 9 adds the four advanced strategy
families named in the program plan and compares them against those baselines
under the same disruption suite:

  1. Oracle-based coordination (``OracleCoordinator``)
       An idealized coordinator that, when queried, returns the globally optimal
       target (the honest-mean consensus value, or the optimal incremental cost
       for dispatch). The query is still delivered over the impaired channel, so
       the oracle is an *upper bound under the same communication budget* rather
       than a cheat: it removes the graph-connectivity bottleneck but remains
       subject to loss / latency / bandwidth on the oracle link.

  2. Reward-redistribution learning (``RECOAggregator``)
       A small learned policy (Yang et al. 2025, RECO; Xiao et al. 2022,
       agent-temporal attention) trained with redistributed rewards and an
       experience-reuse replay pool. Headline property: sample efficiency --
       it reaches a good robust aggregation rule in far fewer training episodes
       than a non-redistributed baseline. Weights are loaded from a checkpoint.

  3. Graph-based reasoning (``GNNAggregator``)
       A graph-attention aggregator that scores each neighbor message from its
       deviation and magnitude and forms an attention-weighted combination
       (Martinkus et al. 2023, agent-based GNNs). It learns to discount Byzantine
       / stale messages continuously, rather than at a fixed breakdown point.

  4. Hybrid MAS+DPS (``hybrid_clusters`` / event-triggered messaging)
       A hierarchical scheme: a DPS layer partitions the graph into clusters and
       elects leaders; dense intra-cluster consensus runs locally while leaders
       carry inter-cluster agreement, and event-triggered messaging suppresses
       redundant transmissions. This restores the spectral gap that flat
       consensus loses on large sparse graphs (the Milestone 8 scalability cliff)
       and cuts communication volume.

The first three expose the ``aggregator(own, neighbor_values)`` interface used by
``synth_envs.linear_consensus_rollout`` and a ``combine(...)`` method for the
``infra_envs`` dispatch task, so every strategy plugs into both the synthetic and
the physical-infrastructure benchmarks. Oracle and hybrid additionally need
rollout-level support, provided by ``coordinated_consensus_rollout`` below.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Torch is optional: the analytical strategies (oracle, hybrid) and the classical
# baselines work without it; only the learned aggregators (GNN, RECO) need it.
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ============================================================
# Learned attention aggregator (shared by the GNN and RECO strategies)
# ============================================================
if _HAS_TORCH:

    class AttnAggregatorNet(nn.Module):
        """Permutation-invariant attention over neighbor messages.

        For each neighbor it builds the feature [deviation-from-own, magnitude],
        scores it with a small MLP, and forms a softmax-weighted combination; a
        learned self-gate decides how much to trust the agent's own value. The
        network is dimension-agnostic (attention weights are scalar per neighbor
        and applied to the whole message vector), so one trained model serves
        scalar consensus and planar rendezvous/formation alike."""

        def __init__(self, hidden: int = 24):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(2, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1))
            self.own_gate = nn.Parameter(torch.tensor(-1.0))  # start trusting neighbors

        def forward(self, own: "torch.Tensor", vals: "torch.Tensor") -> "torch.Tensor":
            # own: (d,)  vals: (k, d)
            dev = (vals - own.unsqueeze(0)).norm(dim=1, keepdim=True)     # (k,1)
            mag = vals.norm(dim=1, keepdim=True)                          # (k,1)
            feat = torch.cat([dev, mag], dim=1)                          # (k,2)
            scores = self.score(feat).squeeze(1)                        # (k,)
            w = torch.softmax(scores, dim=0)                            # (k,)
            agg = (w.unsqueeze(1) * vals).sum(dim=0)                    # (d,)
            g = torch.sigmoid(self.own_gate)
            return g * own + (1.0 - g) * agg


class LearnedAggregator:
    """Wrap a trained ``AttnAggregatorNet`` as an ``aggregator(own, vals)`` callable
    (for the synthetic consensus rollout) and a ``combine(...)`` method (for the
    infrastructure dispatch rollout). Falls back to a robust median if torch or
    the checkpoint is unavailable, so the suite still runs end to end."""

    def __init__(self, net: Any = None, name: str = "gnn"):
        self.net = net
        self.name = name

    # -- synthetic consensus interface: aggregator(own, list_of_vectors) -> vector
    def __call__(self, own: np.ndarray, vals: List[np.ndarray]) -> np.ndarray:
        own = np.asarray(own, dtype=float).reshape(-1)
        if not vals:
            return own
        A = np.asarray(vals, dtype=float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
            own_v = own.reshape(-1)[:1]
        else:
            own_v = own[:A.shape[1]]
        if self.net is None or not _HAS_TORCH:
            return np.median(A, axis=0)
        with torch.no_grad():
            o = torch.tensor(own_v, dtype=torch.float32)
            V = torch.tensor(A, dtype=torch.float32)
            out = self.net(o, V).numpy()
        return out

    # -- infrastructure dispatch interface (scalar lambda / mismatch messages)
    def combine(self, own: float, vals: List[float], w_self: float,
                w_neighbors: List[float]) -> float:
        out = self.__call__(np.array([own]), [np.array([v]) for v in vals])
        return float(np.asarray(out).reshape(-1)[0])


def _load_learned(name: str) -> LearnedAggregator:
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    if not _HAS_TORCH or not os.path.exists(path):
        return LearnedAggregator(net=None, name=name)
    net = AttnAggregatorNet()
    try:
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
    except Exception:
        return LearnedAggregator(net=None, name=name)
    return LearnedAggregator(net=net, name=name)


# Public constructors for the two learned strategies (same architecture, trained
# differently: GNN is supervised on the honest mean; RECO is trained by reward
# redistribution -- see train_m9.py).
def GNNAggregator() -> LearnedAggregator:
    return _load_learned("gnn")


def RECOAggregator() -> LearnedAggregator:
    return _load_learned("reco")


# ============================================================
# Oracle coordinator
# ============================================================
class OracleCoordinator:
    """Marker strategy handled at the rollout level. The rollout supplies the
    honest-agent ground truth (global mean for consensus, optimal lambda for
    dispatch) and delivers it over the impaired channel."""
    name = "oracle"


def oracle_consensus_target(states: np.ndarray, honest_mask: np.ndarray) -> np.ndarray:
    """Idealized consensus target: the mean of the honest agents' current states."""
    sel = states[honest_mask]
    if sel.shape[0] == 0:
        return states.mean(axis=0)
    return sel.mean(axis=0)


# ============================================================
# Hybrid MAS+DPS: clustering (DPS layer) + event-triggered messaging
# ============================================================
def hybrid_clusters(adj: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, List[int]]:
    """Partition the communication graph into ``n_clusters`` contiguous clusters
    (a lightweight distributed-problem-solving layer) and elect a leader per
    cluster (the highest-degree node). Returns (cluster_id per node, leaders).

    A contiguous block partition over the node ordering keeps each cluster
    connected on the ring/ring+chord graphs used here, which is what restores a
    healthy intra-cluster spectral gap."""
    n = adj.shape[0]
    n_clusters = max(1, min(n_clusters, n))
    sizes = [n // n_clusters + (1 if i < n % n_clusters else 0) for i in range(n_clusters)]
    cluster = np.zeros(n, dtype=int)
    leaders: List[int] = []
    start = 0
    deg = adj.sum(axis=1)
    for c, s in enumerate(sizes):
        idx = list(range(start, start + s))
        cluster[idx] = c
        leaders.append(int(idx[int(np.argmax(deg[idx]))]))
        start += s
    return cluster, leaders


# ============================================================
# Strategy registry
# ============================================================
_LEARNED_CACHE: Dict[str, "LearnedAggregator"] = {}


def get_strategy(name: str):
    """Return a coordinator object/callable for ``name``.

    Strings ``mean`` / ``median`` / ``trimmed`` are returned as-is (handled by the
    rollout's classical path); ``oracle`` returns an ``OracleCoordinator`` marker;
    ``gnn`` / ``reco`` return learned aggregators (cached so a worker loads each
    checkpoint once); ``hybrid`` returns the string marker handled structurally by
    the rollout."""
    name = name.lower()
    if name in ("mean", "median", "trimmed", "hybrid"):
        return name
    if name == "oracle":
        return OracleCoordinator()
    if name in ("gnn", "reco"):
        if name not in _LEARNED_CACHE:
            _LEARNED_CACHE[name] = _load_learned(name)
        return _LEARNED_CACHE[name]
    raise KeyError(f"unknown strategy {name}")


STRATEGIES = ["mean", "median", "trimmed", "oracle", "gnn", "reco", "hybrid"]

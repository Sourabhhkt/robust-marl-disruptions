"""
Unit tests for the Milestone 9 additions: advanced coordinators, heterogeneous
teams, the coordinated consensus rollout, and the physical-infrastructure
distributed-dispatch benchmark. Run with:

    python tests/test_m9.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import NetworkFaultConfig
import coordinators as C
import coordinated as CO
import infra_envs as IE
from hetero import make_hetero_profile, homogeneous_profile

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


# ---- Metropolis weights are doubly stochastic ----
def test_mh_weights():
    import synth_envs as SE
    rng = np.random.default_rng(0)
    adj = SE.build_graph(8, "ring_plus", rng, p_extra=0.3)
    W = IE.metropolis_weights(adj)
    check("MH rows sum to 1", np.allclose(W.sum(axis=1), 1.0))
    check("MH cols sum to 1", np.allclose(W.sum(axis=0), 1.0))
    check("MH symmetric", np.allclose(W, W.T))


# ---- infra dispatch converges on a clean channel ----
def test_dispatch_clean():
    rec = IE.dispatch_consensus_rollout(NetworkFaultConfig(seed=0), 0, "mean",
                                        domain="abstract", n=8, steps=80, physics=False)
    check("clean lambda consensus", rec["final_lambda_disagreement"] < 1e-4)
    check("clean power balance", rec["final_power_mismatch"] < 5e-2)
    check("clean near-optimal cost", rec["final_cost_gap"] < 5e-3)


# ---- robust combine rejects Byzantine better than mean (dispatch) ----
def test_dispatch_robust():
    kw = dict(byzantine_comm_corrupt_prob=1.0, spoof_scale=2.0)
    def med(comb):
        return np.median([IE.dispatch_consensus_rollout(
            NetworkFaultConfig(seed=s, **kw), s, comb, domain="abstract", n=12,
            steps=70, byzantine_frac=0.33, physics=False)["final_lambda_disagreement"]
            for s in range(6)])
    check("trimmed beats mean under Byzantine (dispatch)", med("trimmed") < med("mean"))


# ---- physics solvers run and respond ----
def test_physics():
    rg = IE.dispatch_consensus_rollout(NetworkFaultConfig(seed=1), 1, "mean",
                                       domain="gas", n=6, steps=60, physics=True)
    rp = IE.dispatch_consensus_rollout(NetworkFaultConfig(seed=1), 1, "mean",
                                       domain="power", n=6, steps=60, physics=True)
    check("gas physics returns violation frac", "phys_violation_frac" in rg)
    check("power physics returns violation frac", "phys_violation_frac" in rp)
    check("clean gas feasible", rg["phys_violation_frac"] < 0.5)


# ---- coordinated rollout: every strategy returns a valid record ----
def test_strategies():
    for strat in C.STRATEGIES:
        st = C.get_strategy(strat)
        rec = CO.coordinated_consensus_rollout(NetworkFaultConfig(seed=0), 0, st,
                                               n=16, d=1, steps=50)
        check(f"{strat}: finite disagreement", np.isfinite(rec["final_disagreement"]))
        check(f"{strat}: msgs counted", rec["msgs_sent"] > 0)


# ---- oracle is Byzantine-immune; learned beats mean under Byzantine ----
def test_byzantine_ranking():
    kw = dict(byzantine_comm_corrupt_prob=1.0, spoof_scale=2.0)
    def med(strat):
        st = C.get_strategy(strat)
        return np.median([CO.coordinated_consensus_rollout(
            NetworkFaultConfig(seed=s, **kw), s, st, n=20, d=1, steps=60,
            byzantine_frac=0.3)["final_disagreement"] for s in range(6)])
    mean_v, med_v, orc_v = med("mean"), med("median"), med("oracle")
    check("median beats mean under Byzantine", med_v < mean_v)
    check("oracle is near-perfect under Byzantine", orc_v < 1e-6)
    if C.get_strategy("gnn").net is not None:
        check("trained GNN beats median under Byzantine", med("gnn") < med_v)


# ---- hybrid cuts communication volume vs flat ----
def test_hybrid_comm():
    # On the default (denser) ring+chords graph the hybrid restricts inter-cluster
    # links to leaders and event-triggers redundant sends, cutting message volume.
    flat = CO.coordinated_consensus_rollout(NetworkFaultConfig(seed=0), 0, "mean",
                                            n=40, d=1, steps=60, topology="ring_plus")
    hyb = CO.coordinated_consensus_rollout(NetworkFaultConfig(seed=0), 0, "hybrid",
                                           n=40, d=1, steps=60, topology="ring_plus",
                                           n_clusters=6, event_trigger=0.01)
    check("hybrid + event-trigger sends fewer messages than flat",
          hyb["msgs_sent"] < flat["msgs_sent"])


# ---- heterogeneity profile shapes ----
def test_hetero():
    prof = make_hetero_profile(20, 0, level=1.0)
    check("hetero gain_scale shape", prof.gain_scale.shape == (20,))
    check("hetero reliability in [0,1]", np.all((prof.reliability >= 0) & (prof.reliability <= 1)))
    hom = homogeneous_profile(20)
    check("homogeneous gains all 1", np.allclose(hom.gain_scale, 1.0))


if __name__ == "__main__":
    for fn in [test_mh_weights, test_dispatch_clean, test_dispatch_robust, test_physics,
               test_strategies, test_byzantine_ranking, test_hybrid_comm, test_hetero]:
        print(f"\n== {fn.__name__} ==")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)

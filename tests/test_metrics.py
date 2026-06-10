"""Unit tests for control-theoretic metrics and the synthetic environments."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics as M
from core import NetworkFaultConfig
from baselines import run_task


def test_disagreement_and_descent():
    # geometric contraction toward the mean
    T, n = 30, 5
    X = np.zeros((T, n))
    X[0] = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    for t in range(1, T):
        X[t] = 0.7 * X[t - 1] + 0.3 * X[t - 1].mean()
    V = M.disagreement(X)
    assert V[0] > V[-1]
    assert M.monotone_decrease_fraction(V) == 1.0
    cr = M.convergence_rate(V)
    assert 0.0 < cr["rho"] < 1.0
    assert cr["r2"] > 0.95


def test_lambda2_ring_vs_complete():
    import synth_envs as SE
    rng = np.random.default_rng(0)
    ring = SE.build_graph(10, "ring", rng)
    comp = SE.build_graph(10, "complete", rng)
    assert M.laplacian_lambda2(comp) > M.laplacian_lambda2(ring) > 0.0


def test_iae_ise():
    e = np.array([1.0, -2.0, 0.5])
    assert abs(M.iae(e) - 3.5) < 1e-9
    assert abs(M.ise(e) - (1 + 4 + 0.25)) < 1e-9


def test_time_to_threshold():
    V = np.array([10.0, 5.0, 1.0, 0.1])
    assert M.time_to_threshold(V, 0.5) == 3
    assert M.time_to_threshold(V, 1e-6) is None


def test_clean_channel_converges():
    # average consensus on a clean channel reaches agreement; median is slower but converges.
    cfg = NetworkFaultConfig(seed=0)
    rec = run_task("consensus_mean", cfg, seed=0)
    assert rec["final_disagreement"] < 1e-4
    assert rec["lyap_rho"] < 1.0
    assert rec["converged"] == 1.0


def test_total_loss_breaks_consensus():
    cfg = NetworkFaultConfig(seed=0, msg_drop_prob=1.0)
    rec = run_task("consensus_mean", cfg, seed=0)
    assert rec["final_disagreement"] > 1.0      # cannot converge with no messages
    assert rec["delivery_rate"] == 0.0


def test_median_beats_mean_under_byzantine():
    # with Byzantine liars, robust aggregation should keep disagreement far lower.
    f_mean, f_med = [], []
    for s in range(5):
        cfg = NetworkFaultConfig(seed=s, byzantine_comm_corrupt_prob=1.0, spoof_scale=2.0)
        f_mean.append(run_task("consensus_mean", cfg, seed=s, env_overrides={"byzantine_frac": 0.25})["final_disagreement"])
        f_med.append(run_task("consensus_median", cfg, seed=s, env_overrides={"byzantine_frac": 0.25})["final_disagreement"])
    assert np.median(f_med) < np.median(f_mean)


def test_rate_distortion_knee():
    # coarser quantization -> worse (or equal) consensus; very low bits fail.
    cfg1 = NetworkFaultConfig(seed=0, bandwidth_bits=1, quant_clip=(-3, 3))
    cfg8 = NetworkFaultConfig(seed=0, bandwidth_bits=8, quant_clip=(-3, 3))
    r1 = run_task("consensus_mean", cfg1, seed=0)["final_disagreement"]
    r8 = run_task("consensus_mean", cfg8, seed=0)["final_disagreement"]
    assert r1 > r8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} metric tests passed")

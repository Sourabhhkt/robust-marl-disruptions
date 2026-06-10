"""
Unit tests for the hardened communication-fault core.

Run:  python -m pytest tests/ -q     (or)   python tests/test_core.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import NetworkChannel, FaultModel, NetworkFaultConfig, DELIVERED_STATUSES


def test_perfect_channel_delivers_exactly():
    cfg = NetworkFaultConfig(seed=0)
    ch = NetworkChannel(cfg); ch.reset(["a", "b"])
    p = np.array([1.0, 2.0, 3.0])
    ch.send("a", "b", p, t=0, agents=["a", "b"])
    msg, meta = ch.recv("b", t=0, dim=3, agents=["a", "b"])
    assert meta["status"] == "delivered"
    assert np.allclose(msg, p)
    assert meta["age"] == 0
    assert np.allclose(meta["src_payload"], p)
    # queue is now empty -> a second recv reports "none", not a stale copy
    msg2, meta2 = ch.recv("b", t=1, dim=3, agents=["a", "b"])
    assert meta2["status"] == "none"
    assert np.allclose(msg2, 0.0)


def test_drop_consumes_message_and_bounds_queue():
    # B1: a dropped message must be popped (not left to grow the queue) and is
    # distinguishable from "no message".
    cfg = NetworkFaultConfig(seed=3, msg_drop_prob=1.0)
    ch = NetworkChannel(cfg); ch.reset(["a", "b"])
    for t in range(50):
        ch.send("a", "b", np.array([float(t)]), t=t, agents=["a", "b"])
        msg, meta = ch.recv("b", t=t, dim=1, agents=["a", "b"])
        assert meta["status"] == "dropped"
        assert np.allclose(msg, 0.0)
    # all messages consumed despite 100% drop
    assert len(ch._q["b"]) == 0


def test_latency_and_real_age():
    # B5: age reflects t - creation time, including latency.
    cfg = NetworkFaultConfig(seed=1, base_latency_steps=3)
    ch = NetworkChannel(cfg); ch.reset(["a", "b"])
    ch.send("a", "b", np.array([5.0]), t=0, agents=["a", "b"])
    # not deliverable before t=3
    _, meta_early = ch.recv("b", t=2, dim=1, agents=["a", "b"])
    assert meta_early["status"] == "none"
    msg, meta = ch.recv("b", t=3, dim=1, agents=["a", "b"])
    assert meta["status"] == "delivered"
    assert meta["age"] == 3
    assert np.allclose(msg, 5.0)


def test_quantization_distortion_monotone():
    # sender-anchored distortion grows as bits shrink.
    p = np.linspace(-1, 1, 16)

    def delivered_mse(bits):
        cfg = NetworkFaultConfig(seed=0, bandwidth_bits=bits, quant_clip=(-1, 1))
        ch = NetworkChannel(cfg); ch.reset(["a", "b"])
        ch.send("a", "b", p, t=0, agents=["a", "b"])
        msg, meta = ch.recv("b", t=0, dim=16, agents=["a", "b"])
        return float(np.mean((meta["src_payload"] - msg) ** 2))

    mse_hi = delivered_mse(8)
    mse_lo = delivered_mse(2)
    assert mse_lo > mse_hi >= 0.0


def test_recv_all_per_source():
    # recv_all delivers the freshest message from each source.
    cfg = NetworkFaultConfig(seed=0)
    ch = NetworkChannel(cfg); ch.reset(["a", "b", "c"])
    ch.send("a", "c", np.array([1.0]), t=0, agents=["a", "b", "c"])
    ch.send("b", "c", np.array([2.0]), t=0, agents=["a", "b", "c"])
    out = ch.recv_all("c", t=0, dim=1, agents=["a", "b", "c"])
    assert set(out.keys()) == {"a", "b"}
    assert np.allclose(out["a"][0], 1.0)
    assert np.allclose(out["b"][0], 2.0)
    # consumed
    assert len(ch._q["c"]) == 0


def test_partition_does_not_consume():
    cfg = NetworkFaultConfig(seed=0, partitioned_agents_fn=lambda t, ag: {"b"})
    ch = NetworkChannel(cfg); ch.reset(["a", "b"])
    ch.send("a", "b", np.array([7.0]), t=0, agents=["a", "b"])
    msg, meta = ch.recv("b", t=0, dim=1, agents=["a", "b"])
    assert meta["status"] == "partitioned"
    assert len(ch._q["b"]) == 1  # retained for after heal


def test_crash_sampled_once_per_step():
    # B2: calling transform_observation and transform_action in the same step
    # must not double-sample crash onset.
    cfg = NetworkFaultConfig(seed=2, crash_prob=0.5, crash_duration=2)
    fm = FaultModel(cfg); fm.reset(["a"])
    # Reference: only observation path sampled, one draw per step.
    rng_draws = []
    fm_ref = FaultModel(cfg); fm_ref.reset(["a"])
    crashed_double = 0
    crashed_single = 0
    fm2 = FaultModel(cfg); fm2.reset(["a"])
    for t in range(200):
        # double-call path (obs + action) within same t
        fm.transform_observation("a", np.zeros(2), t)
        fm.transform_action("a", 0, t, noop_action_fn=lambda: 0, random_action_fn=lambda: 1)
        if fm._is_crashed("a", t):
            crashed_double += 1
        # single-call path
        fm2.transform_observation("a", np.zeros(2), t)
        if fm2._is_crashed("a", t):
            crashed_single += 1
    # identical crash trajectory regardless of how many calls per step
    assert crashed_double == crashed_single


def test_independent_rng_streams():
    # channel and fault model must not share a stream.
    cfg = NetworkFaultConfig(seed=5)
    ch = NetworkChannel(cfg)
    fm = FaultModel(cfg)
    a = ch.rng.random(20)
    b = fm.rng.random(20)
    assert not np.allclose(a, b)


def test_byzantine_corruption_on_send():
    cfg = NetworkFaultConfig(seed=0, byzantine_agents={"a"}, byzantine_comm_corrupt_prob=1.0, spoof_scale=1.0)
    ch = NetworkChannel(cfg); ch.reset(["a", "b"])
    p = np.array([10.0, 10.0, 10.0])
    ch.send("a", "b", p, t=0, agents=["a", "b"], allow_byzantine_corrupt=True)
    msg, meta = ch.recv("b", t=0, dim=3, agents=["a", "b"])
    assert meta["status"] in DELIVERED_STATUSES
    assert not np.allclose(msg, p)  # corrupted


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} core tests passed")

"""
Heterogeneous-team model for Milestone 9.

The Milestone 8 study used homogeneous teams: every agent had the same update
gain, the same link quality, and the same message precision. The Milestone 9
exit criteria call for a comparison of scalability, resilience, and adaptability
"across heterogeneous agent teams", so we equip the coordinated rollout with a
per-agent profile that varies four capability axes:

  - gain_scale     : per-agent update-gain multiplier (fast vs. sluggish actuators
                     / compute budgets).
  - reliability    : per-agent probability of successfully transmitting in a step
                     (good vs. degraded radios / links), on top of the channel's
                     own impairments.
  - precision_bits : per-agent message precision in bits (full-rate vs.
                     bandwidth-starved nodes); None / 0 means full precision.
  - is_leader      : role flag used by the hybrid MAS+DPS coordinator.

``level`` in [0, 1] scales the spread: 0 reproduces a homogeneous team, 1 a
strongly heterogeneous one. Adaptive coordinators (oracle, graph-attention,
reward-redistribution) can exploit this structure -- down-weighting unreliable or
low-precision neighbors -- whereas a plain average cannot, which is exactly the
adaptability dimension the milestone asks us to quantify.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class HeteroProfile:
    gain_scale: np.ndarray       # (n,) update-gain multiplier, >0
    reliability: np.ndarray      # (n,) transmit success probability in [0,1]
    precision_bits: np.ndarray   # (n,) bits/message; 0 == full precision
    is_leader: np.ndarray        # (n,) bool

    @property
    def n(self) -> int:
        return int(self.gain_scale.shape[0])


def make_hetero_profile(n: int, seed: int, level: float = 1.0,
                        n_leaders: Optional[int] = None) -> HeteroProfile:
    """Sample a heterogeneous-team profile. ``level`` interpolates from a
    homogeneous team (0) to a strongly heterogeneous one (1)."""
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(8)[7])
    level = float(np.clip(level, 0.0, 1.0))

    # capability: gains spread multiplicatively around 1 (some agents 2x faster)
    gain_scale = np.exp(level * rng.uniform(-0.7, 0.7, size=n))

    # link reliability: a fraction of agents have degraded radios
    reliability = np.ones(n)
    degraded = rng.random(n) < (0.4 * level)
    reliability[degraded] = 1.0 - level * rng.uniform(0.3, 0.8, size=int(degraded.sum()))

    # precision: a fraction of agents are bandwidth-starved (2-4 bits), rest full
    precision_bits = np.zeros(n)             # 0 == full precision
    starved = rng.random(n) < (0.4 * level)
    precision_bits[starved] = rng.integers(2, 5, size=int(starved.sum()))

    is_leader = np.zeros(n, dtype=bool)
    if n_leaders:
        is_leader[:max(1, n_leaders)] = True

    return HeteroProfile(gain_scale=gain_scale, reliability=reliability,
                         precision_bits=precision_bits, is_leader=is_leader)


def homogeneous_profile(n: int) -> HeteroProfile:
    return HeteroProfile(gain_scale=np.ones(n), reliability=np.ones(n),
                         precision_bits=np.zeros(n), is_leader=np.zeros(n, dtype=bool))

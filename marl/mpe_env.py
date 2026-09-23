"""
PettingZoo MPE wrapper as a cooperative MARL env with the disruption applied to
the inter-agent observation slice (the other agents' relative positions, which
are the implicit messages an agent receives about its teammates). Discrete
actions, homogeneous agents, shared cooperative reward -- so the discrete-action
methods (QMIX, and discrete IPPO/MAPPO) apply naturally, matching the M9 rule
that discrete value methods belong on discrete-action tasks.

The impairment mirrors the disruption families on that slice (per-element drop /
quantization / spoofing noise), the same observation-slice augmentation used by
the Milestone 7/8 wrapper for the wrapped real benchmarks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from core import NetworkFaultConfig, quantize_uniform


# inter-agent observation slice (start, length) for each MPE scenario.
_COMM_SLICE = {"mpe_spread": None, "mpe_reference": None, "mpe_speaker": None}


def _make_pz(name: str, n: int):
    if name == "mpe_spread":
        from mpe2 import simple_spread_v3
        return simple_spread_v3.parallel_env(N=n, max_cycles=25, continuous_actions=False), \
            (4 + 2 * n, 2 * (n - 1))      # other-agent relative positions
    if name == "mpe_reference":
        from mpe2 import simple_reference_v3
        # obs = vel(2) + landmarks(6) + goal(3) + comm(10): the trailing 10 dims
        # are the other agent's message -- the slice the disruption impairs.
        return simple_reference_v3.parallel_env(local_ratio=0.5, max_cycles=25,
                                                continuous_actions=False), (11, 10)
    raise KeyError(f"unknown MPE env {name}")


class MPEMARLEnv:
    def __init__(self, name: str = "mpe_spread", n: int = 3, regime: str = "clean"):
        self.name = name
        self.n_req = n
        self.regime = regime
        self._pz, self._slice = _make_pz(name, n)
        obs, _ = self._pz.reset(seed=0)
        self.agent_ids = list(self._pz.agents)
        self.n_agents = len(self.agent_ids)
        a0 = self.agent_ids[0]
        self.obs_dim = int(obs[a0].shape[0])
        self.n_actions = int(self._pz.action_space(a0).n)
        self.continuous = False
        self.act_dim = 1
        self.state_dim = self.obs_dim * self.n_agents
        self._rng = np.random.default_rng(0)

    def _impair(self, obs_arr: np.ndarray) -> np.ndarray:
        if self._slice is None or self.cfg is None:
            return obs_arr
        s, L = self._slice
        seg = obs_arr[:, s:s + L].copy()
        if self.cfg.msg_drop_prob > 0:
            mask = self._rng.random(seg.shape) < self.cfg.msg_drop_prob
            seg[mask] = 0.0
        if self.cfg.bandwidth_bits:
            seg = quantize_uniform(seg, int(self.cfg.bandwidth_bits), self.cfg.quant_clip)
        if self.cfg.spoof_prob > 0:
            mask = self._rng.random(seg.shape[0]) < self.cfg.spoof_prob
            seg[mask] += self._rng.normal(0, self.cfg.spoof_scale, size=(int(mask.sum()), L))
        obs_arr[:, s:s + L] = seg
        return obs_arr

    def _stack(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        arr = np.stack([obs[a].astype(np.float32) for a in self.agent_ids])
        return self._impair(arr)

    def reset(self, seed: int = 0, cfg: Optional[NetworkFaultConfig] = None,
              byz_frac: Optional[float] = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        if cfg is not None:
            self.cfg = cfg
        elif self.regime == "disrupt":
            from .envs import sample_disruption
            self.cfg, _ = sample_disruption(self._rng)
        else:
            self.cfg = NetworkFaultConfig()
        obs, _ = self._pz.reset(seed=seed)
        self.agent_ids = list(self._pz.agents)
        self._last = self._stack(obs)
        return self._last

    def step(self, actions) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        actions = np.asarray(actions).reshape(-1).astype(int)
        act_dict = {a: int(actions[i]) for i, a in enumerate(self.agent_ids)}
        obs, rew, term, trunc, info = self._pz.step(act_dict)
        shared = float(np.mean([rew[a] for a in self.agent_ids])) if rew else 0.0
        done = (not self._pz.agents) or all(term.values()) or all(trunc.values())
        if not done:
            self._last = self._stack(obs)
        return self._last, shared, done, {"reward": shared}

    def state(self) -> np.ndarray:
        return self._last.reshape(-1).astype(np.float32)

    def alive_mask(self) -> np.ndarray:
        return np.ones(self.n_agents, dtype=bool)

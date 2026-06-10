# adapter.py
#
# Benchmark-specific translation layer for the universal communication-fault
# wrapper. Adapters tell the wrapper how to interpret actions/messages/
# observations for a given benchmark.
#
# This module contains only the adapters that are *validated and used* by the
# the validated experiments (PettingZoo MPE, WNTR via obs-slice). Untested
# skeleton adapters for SMAC / Flatland / Overcooked / MAPF / RoboCup / FRODO
# have been moved to ``contrib/adapter_experimental.py`` and are clearly marked
# as not validated.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np


class UniversalBenchmarkAdapter:
    """
    Supports both:
      1) explicit communication benchmarks
      2) implicit communication via observation/state slices
    """

    def agents(self, env) -> Sequence[str]:
        raise NotImplementedError

    def split_action(self, agent: str, action: Any) -> Tuple[Any, Optional[np.ndarray]]:
        return split_action_augmented(action)

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        raise NotImplementedError

    def extract_outgoing_message(
        self,
        env,
        agent: str,
        obs: Any,
        env_action: Any,
        explicit_msg_out: Optional[np.ndarray],
    ) -> np.ndarray:
        if explicit_msg_out is None:
            return np.zeros((self.message_dim(env, agent, obs),), dtype=float)
        return explicit_msg_out.astype(float, copy=False)

    def inject_incoming_message(
        self,
        env,
        agent: str,
        obs: Any,
        msg_in: np.ndarray,
    ) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode="dict", key="msg")

    def noop_action(self, env, agent: str) -> Any:
        space = env.action_space(agent)
        try:
            return space({})
        except Exception:
            pass
        if hasattr(space, "n"):
            return 0
        if hasattr(space, "shape") and space.shape is not None:
            return np.zeros(space.shape, dtype=np.float32)
        return 0

    def random_action(self, env, agent: str) -> Any:
        return env.action_space(agent).sample()


# ============================================================
# Common helpers
# ============================================================

def default_agents(n: int, prefix: str = "agent_") -> List[str]:
    return [f"{prefix}{i}" for i in range(n)]


def coerce_1d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array-like obs/message, got shape={arr.shape}")
    return arr


def inject_msg_into_obs(
    obs: Any,
    msg_in: np.ndarray,
    mode: str = "concat",
    key: str = "msg",
) -> Any:
    """
    mode:
      - "concat": obs must be 1D numeric array; returns np.concatenate([obs, msg_in])
      - "dict":   obs becomes/extends dict with obs[key] = msg_in
      - "tuple":  returns (obs, msg_in)
      - "noop":   returns obs unchanged
      - "overwrite_tail": overwrite the last len(msg_in) entries of a 1D obs
    """
    if mode == "noop":
        return obs
    if mode == "tuple":
        return (obs, msg_in)

    if mode == "dict":
        if isinstance(obs, dict):
            out = dict(obs)
            out[key] = msg_in
            return out
        return {"observation": obs, key: msg_in}

    if mode == "overwrite_tail":
        base = np.asarray(obs, dtype=float).copy().reshape(-1)
        k = min(msg_in.shape[0], base.shape[0])
        base[-k:] = np.asarray(msg_in, dtype=float).reshape(-1)[:k]
        return base

    # concat
    base = coerce_1d_float(obs)
    return np.concatenate([base, np.asarray(msg_in, dtype=float).reshape(-1)], axis=0)


def split_action_augmented(action: Any) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Convention: action can be either:
      - env_action
      - (env_action, msg_out)
    """
    if isinstance(action, tuple) and len(action) == 2:
        env_action, msg_out = action
        if msg_out is None:
            return env_action, None
        return env_action, coerce_1d_float(msg_out)
    return action, None


# ============================================================
# ObsSliceCommAdapter: communication is a slice of the observation
# ============================================================

@dataclass
class ObsSliceCommAdapter(UniversalBenchmarkAdapter):
    """
    Communication is represented as a slice of each agent's observation vector.
    Used for benchmarks without a native message channel (WNTR) and
    for MPE where we designate part of the observation as the broadcast message.
    """
    agents_list_fn: Callable[[Any], Sequence[str]]
    slice_fn: Callable[[Any], np.ndarray]
    replace_fn: Callable[[Any, np.ndarray], Any]

    def agents(self, env) -> Sequence[str]:
        return list(self.agents_list_fn(env))

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        if obs is None:
            raise ValueError("ObsSliceCommAdapter.message_dim needs obs")
        return int(np.asarray(self.slice_fn(obs)).reshape(-1).shape[0])

    def extract_outgoing_message(
        self,
        env,
        agent: str,
        obs: Any,
        env_action: Any,
        explicit_msg_out: Optional[np.ndarray],
    ) -> np.ndarray:
        return np.asarray(self.slice_fn(obs), dtype=float).reshape(-1)

    def inject_incoming_message(
        self,
        env,
        agent: str,
        obs: Any,
        msg_in: np.ndarray,
    ) -> Any:
        return self.replace_fn(obs, msg_in)


# ============================================================
# DictMAAdapter: generic dict-based multi-agent benchmark
# ============================================================

@dataclass
class DictMAAdapter(UniversalBenchmarkAdapter):
    """
    Works when env.reset()/env.step() use dicts keyed by agent_id and the
    message is provided explicitly via the action ``(env_action, msg_out)``.
    """
    agents_list_fn: Callable[[Any], Sequence[str]]
    msg_dim: int
    obs_inject_mode: str = "dict"      # "dict" | "concat" | "tuple" | "noop"
    obs_msg_key: str = "msg"

    def agents(self, env) -> Sequence[str]:
        return list(self.agents_list_fn(env))

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        return int(self.msg_dim)

    def inject_incoming_message(self, env, agent: str, obs: Any, msg_in: np.ndarray) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple
import numpy as np


# ============================================================
# Helper: uniform quantization (bandwidth/precision emulation)
# ============================================================
def quantize_uniform(x: np.ndarray, bits: int, clip: Tuple[float, float]) -> np.ndarray:
    """
    Uniform quantization to emulate bandwidth/precision limits.
    bits <= 0 => all zeros
    """
    x = np.asarray(x, dtype=float)
    if bits <= 0:
        return np.zeros_like(x)

    lo, hi = clip
    if not (hi > lo):
        return np.zeros_like(x)

    x = np.clip(x, lo, hi)
    levels = (1 << bits) - 1
    if levels <= 0:
        return np.zeros_like(x)

    q = np.round((x - lo) * levels / (hi - lo))
    return lo + q * (hi - lo) / levels


# ============================================================
# Config: channel + faults (very close to your PettingZoo cfg)
# ============================================================
@dataclass
class NetworkFaultConfig:
    seed: int = 1

    # -----------------------
    # Network / channel model
    # -----------------------
    msg_drop_prob: float = 0.0             # per-message drop (receiver-side)
    msg_duplicate_prob: float = 0.0        # duplicate enqueue
    msg_reorder_prob: float = 0.0          # probabilistic swap in receiver queue

    base_latency_steps: int = 0            # fixed base latency
    jitter_steps: int = 0                  # +/- jitter (integer)

    ttl_steps: Optional[int] = None        # if set, drop delivered msg older than ttl

    bandwidth_bits: Optional[int] = None   # quantize payload entries
    quant_clip: Tuple[float, float] = (-1.0, 1.0)
    max_msg_dims: Optional[int] = None     # keep only first k dims

    # Partitions/topology: if receiver is partitioned, it receives zeros
    partitioned_agents_fn: Optional[Callable[[int, Sequence[str]], Set[str]]] = None

    # Jamming: if receiver jammed, drop with jam_drop_prob
    jammed_agents_fn: Optional[Callable[[int, Sequence[str]], Set[str]]] = None
    jam_drop_prob: float = 1.0

    # Spoofing/corruption: replace delivered msg with noise
    spoof_prob: float = 0.0
    spoof_scale: float = 1.0

    # Replay: deliver an older deliverable msg without popping it (or pick random deliverable)
    replay_prob: float = 0.0

    # -----------------------
    # Fault model (agents)
    # -----------------------
    crash_prob: float = 0.0
    crash_duration: int = 5

    sensor_noise_prob: float = 0.0
    sensor_noise_scale: float = 0.0

    actuator_fault_prob: float = 0.0
    actuator_fault_mode: str = "random"    # "random" or "noop"

    byzantine_agents: Optional[Set[str]] = None
    byzantine_comm_corrupt_prob: float = 0.0
    byzantine_action_prob: float = 0.0


# ============================================================
# Core 1: NetworkChannel (benchmark-agnostic)
# ============================================================
class NetworkChannel:
    """
    Benchmark-agnostic message channel with impairments.

    Usage pattern in wrappers/harnesses:
        channel = NetworkChannel(cfg)
        channel.reset(agents)

        channel.send(src, dst, payload, t, agents=agents)
        msg = channel.recv(dst, t, dim=msg_dim, agents=agents)
    """
    def __init__(self, cfg: NetworkFaultConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._q: Dict[str, list[tuple[int, int, np.ndarray]]] = {}  # dst -> [(deliver_t, created_t, payload)]
        self._agents_cache: list[str] = []

    def reset(self, agents: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._q.clear()
        self._agents_cache = list(agents) if agents is not None else []

        if agents is not None:
            for a in agents:
                self._q.setdefault(a, [])

    def _ensure_agent(self, agent: str):
        if agent not in self._q:
            self._q[agent] = []

    def _delay_steps(self) -> int:
        base = int(self.cfg.base_latency_steps)
        jit = int(self.cfg.jitter_steps)
        if jit > 0:
            base = base + int(self.rng.integers(-jit, jit + 1))
        return max(0, base)

    def _apply_bandwidth_cap(self, msg: np.ndarray) -> np.ndarray:
        msg = np.asarray(msg, dtype=float)

        if self.cfg.max_msg_dims is not None:
            k = int(self.cfg.max_msg_dims)
            if k < msg.shape[0]:
                out = msg.copy()
                out[k:] = 0.0
                msg = out

        if self.cfg.bandwidth_bits is not None:
            msg = quantize_uniform(msg, int(self.cfg.bandwidth_bits), self.cfg.quant_clip)

        return msg

    def _is_partitioned(self, t: int, receiver: str, agents: Sequence[str]) -> bool:
        if self.cfg.partitioned_agents_fn is None:
            return False
        return receiver in set(self.cfg.partitioned_agents_fn(t, agents))

    def _is_jammed(self, t: int, receiver: str, agents: Sequence[str]) -> bool:
        if self.cfg.jammed_agents_fn is None:
            return False
        return receiver in set(self.cfg.jammed_agents_fn(t, agents))

    def send(
        self,
        src: str,
        dst: str,
        payload: np.ndarray,
        t: int,
        agents: Optional[Sequence[str]] = None,
        *,
        allow_byzantine_corrupt: bool = False,
    ):
        """
        Enqueue payload to dst with latency/jitter, duplication, reordering.
        NOTE: Drop/jam/partition decisions are applied at recv-time (receiver-side model).
        """
        self._ensure_agent(dst)
        payload = np.asarray(payload, dtype=float).copy()

        # Optional: corrupt outgoing payload if src is byzantine (if you want it here).
        # Many people prefer doing byzantine comm in the wrapper before calling send(),
        # but this hook lets you keep it centralized if you want.
        if allow_byzantine_corrupt and (self.cfg.byzantine_agents is not None) and (src in self.cfg.byzantine_agents):
            if self.cfg.byzantine_comm_corrupt_prob > 0 and (self.rng.random() < self.cfg.byzantine_comm_corrupt_prob):
                payload = self.rng.normal(0.0, self.cfg.spoof_scale, size=payload.shape).astype(float)

        dt = t + self._delay_steps()
        ct = t

        q = self._q[dst]
        q.append((dt, ct, payload))

        # duplication
        if self.cfg.msg_duplicate_prob > 0 and (self.rng.random() < self.cfg.msg_duplicate_prob):
            q.append((dt, ct, payload.copy()))

        # reordering (local swap)
        if self.cfg.msg_reorder_prob > 0 and len(q) >= 2 and (self.rng.random() < self.cfg.msg_reorder_prob):
            q[-1], q[-2] = q[-2], q[-1]

    def recv(
        self,
        receiver: str,
        t: int,
        dim: int,
        agents: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Deliver a message to receiver at time t:
          - partitions/jamming/drop
          - FIFO deliver earliest deliverable, or replay attack
          - TTL stale drop
          - spoofing
          - bandwidth caps
        Returns a vector of shape (dim,).
        """
        self._ensure_agent(receiver)
        agents_list = list(agents) if agents is not None else (self._agents_cache if self._agents_cache else [receiver])

        # partitions
        if self._is_partitioned(t, receiver, agents_list):
            return np.zeros((dim,), dtype=float)

        # jamming
        if self._is_jammed(t, receiver, agents_list):
            if self.rng.random() < self.cfg.jam_drop_prob:
                return np.zeros((dim,), dtype=float)

        # base drop
        if self.cfg.msg_drop_prob > 0 and (self.rng.random() < self.cfg.msg_drop_prob):
            return np.zeros((dim,), dtype=float)

        q = self._q[receiver]
        deliverable_idx = [i for i, (dt, _, _) in enumerate(q) if dt <= t]
        if not deliverable_idx:
            return np.zeros((dim,), dtype=float)

        # replay: pick a deliverable message but do not remove (or just pick random deliverable)
        if self.cfg.replay_prob > 0 and (self.rng.random() < self.cfg.replay_prob):
            pick = int(self.rng.choice(deliverable_idx))
            dt, ct, payload = q[pick]
        else:
            # FIFO on deliver_time among deliverables
            pick = min(deliverable_idx, key=lambda i: q[i][0])
            dt, ct, payload = q.pop(pick)

        # TTL (stale)
        if self.cfg.ttl_steps is not None:
            if (t - ct) > int(self.cfg.ttl_steps):
                return np.zeros((dim,), dtype=float)

        # spoofing
        if self.cfg.spoof_prob > 0 and (self.rng.random() < self.cfg.spoof_prob):
            payload = self.rng.normal(loc=0.0, scale=self.cfg.spoof_scale, size=(dim,)).astype(float)
        else:
            payload = np.asarray(payload, dtype=float)

        # force dim (pad/trim)
        if payload.shape[0] < dim:
            out = np.zeros((dim,), dtype=float)
            out[: payload.shape[0]] = payload
            payload = out
        elif payload.shape[0] > dim:
            payload = payload[:dim]

        # bandwidth caps applied after delivery
        payload = self._apply_bandwidth_cap(payload)
        return payload.astype(float, copy=False)


# ============================================================
# Core 2: FaultModel (benchmark-agnostic)
# ============================================================
class FaultModel:
    """
    Benchmark-agnostic agent faults:
      - crash-stop / intermittent availability
      - sensor noise on numpy-like observations
      - actuator faults on generic actions (supports:
            * discrete ints (noop=0)
            * numpy actions
            * fallback: use action_sampler if provided)
      - byzantine action overriding (random action)
    """
    def __init__(self, cfg: NetworkFaultConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._crash_until: Dict[str, int] = {}
        self._last_action: Dict[str, Any] = {}

    def reset(self, agents: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._crash_until.clear()
        self._last_action.clear()
        if agents is not None:
            for a in agents:
                self._crash_until[a] = -1

    def _is_crashed(self, agent: str, t: int) -> bool:
        return t < self._crash_until.get(agent, -1)

    def _maybe_start_crash(self, agent: str, t: int):
        if self._is_crashed(agent, t):
            return
        if self.cfg.crash_prob > 0 and (self.rng.random() < self.cfg.crash_prob):
            self._crash_until[agent] = t + max(1, int(self.cfg.crash_duration))

    def transform_observation(self, agent: str, obs: Any, t: int) -> Any:
        self._maybe_start_crash(agent, t)

        if self._is_crashed(agent, t):
            try:
                arr = np.asarray(obs)
                return np.zeros_like(arr)
            except Exception:
                return obs

        if self.cfg.sensor_noise_prob > 0 and (self.rng.random() < self.cfg.sensor_noise_prob):
            if self.cfg.sensor_noise_scale > 0:
                try:
                    arr = np.asarray(obs, dtype=float)
                    noise = self.rng.uniform(
                        low=-self.cfg.sensor_noise_scale,
                        high=self.cfg.sensor_noise_scale,
                        size=arr.shape,
                    )
                    return arr + noise
                except Exception:
                    return obs

        return obs

    def transform_action(
        self,
        agent: str,
        action: Any,
        t: int,
        *,
        noop_action_fn: Callable[[], Any],
        random_action_fn: Callable[[], Any],
    ) -> Any:
        self._maybe_start_crash(agent, t)

        if self._is_crashed(agent, t):
            if self.cfg.actuator_fault_mode == "noop":
                return noop_action_fn()
            if agent in self._last_action:
                return self._last_action[agent]
            return random_action_fn()

        byz = self.cfg.byzantine_agents or set()

        if agent in byz and self.cfg.byzantine_action_prob > 0 and (self.rng.random() < self.cfg.byzantine_action_prob):
            action = random_action_fn()

        if self.cfg.actuator_fault_prob > 0 and (self.rng.random() < self.cfg.actuator_fault_prob):
            if self.cfg.actuator_fault_mode == "random":
                action = random_action_fn()
            elif self.cfg.actuator_fault_mode == "noop":
                action = noop_action_fn()

        self._last_action[agent] = action
        return action

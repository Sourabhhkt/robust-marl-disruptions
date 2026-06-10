from __future__ import annotations

from dataclasses import dataclass
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


def _spawn_rng(seed: int, child: int, n_children: int = 2) -> np.random.Generator:
    """
    Build an independent random stream from a base seed.

    The communication channel and the agent fault model must use *independent*
    streams so that, e.g., a message-drop sweep and a crash sweep do not share
    correlated random draws. ``SeedSequence(seed).spawn(n)`` yields ``n``
    statistically independent child sequences; ``child`` selects which one.
    """
    children = np.random.SeedSequence(int(seed)).spawn(int(n_children))
    return np.random.default_rng(children[int(child)])


# ============================================================
# Config: channel + faults
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

    # Replay: re-deliver an older deliverable msg without popping it
    replay_prob: float = 0.0

    # Safety bound on per-receiver queue length (prevents unbounded growth
    # under heavy latency/partition where messages accumulate). Oldest are
    # dropped first when the bound is exceeded.
    max_queue_len: int = 4096

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


# Possible per-receive statuses returned in recv() metadata.
RECV_STATUSES = (
    "none",        # no deliverable message in queue
    "delivered",   # a fresh message was delivered
    "replayed",    # an old (already-delivered) message was re-delivered
    "spoofed",     # delivered payload was replaced by adversarial noise
    "dropped",     # message was popped then lost (packet loss)
    "jammed",      # message popped then lost due to jamming
    "partitioned", # receiver isolated; no delivery this step
    "stale",       # message exceeded its time-to-live
)
# Statuses for which a (nonzero) real payload reaches the receiver.
DELIVERED_STATUSES = ("delivered", "replayed", "spoofed")


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
        msg, meta = channel.recv(dst, t, dim=msg_dim, agents=agents)

    ``recv`` returns ``(payload, meta)`` where ``meta`` carries:
        - status:      one of RECV_STATUSES
        - age:         integer message age (t - creation_time), 0 if none
        - src_payload: the payload as sent by the source (pre-impairment),
                       or None if no message was delivered. This enables a
                       *sender-anchored* distortion metric (compare what was
                       delivered against what was actually sent).
    """

    def __init__(self, cfg: NetworkFaultConfig):
        self.cfg = cfg
        self.rng = _spawn_rng(cfg.seed, child=0)
        self._q: Dict[str, list[tuple[int, int, np.ndarray]]] = {}  # dst -> [(deliver_t, created_t, payload)]
        self._agents_cache: list[str] = []

    def reset(self, agents: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        base_seed = self.cfg.seed if seed is None else seed
        self.rng = _spawn_rng(base_seed, child=0)
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

        # Optionally corrupt outgoing payload if src is byzantine.
        if allow_byzantine_corrupt and (self.cfg.byzantine_agents is not None) and (src in self.cfg.byzantine_agents):
            if self.cfg.byzantine_comm_corrupt_prob > 0 and (self.rng.random() < self.cfg.byzantine_comm_corrupt_prob):
                payload = self.rng.normal(0.0, self.cfg.spoof_scale, size=payload.shape).astype(float)

        dt = t + self._delay_steps()
        ct = t

        q = self._q[dst]
        q.append((dt, ct, src, payload))

        # duplication
        if self.cfg.msg_duplicate_prob > 0 and (self.rng.random() < self.cfg.msg_duplicate_prob):
            q.append((dt, ct, src, payload.copy()))

        # reordering (local swap of the two most recent enqueues)
        if self.cfg.msg_reorder_prob > 0 and len(q) >= 2 and (self.rng.random() < self.cfg.msg_reorder_prob):
            q[-1], q[-2] = q[-2], q[-1]

        # bound queue growth (drop oldest first)
        max_len = int(self.cfg.max_queue_len)
        if max_len > 0 and len(q) > max_len:
            del q[: len(q) - max_len]

    def recv(
        self,
        receiver: str,
        t: int,
        dim: int,
        agents: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Deliver a message to ``receiver`` at time ``t``. Returns ``(payload, meta)``.

        Delivery model (receiver side):
          1. If the receiver is partitioned, nothing is delivered and queued
             messages are retained (they may arrive once the partition heals).
          2. Otherwise the earliest deliverable message is *popped* (FIFO on
             deliver-time). A replay attack instead *peeks* a random deliverable
             message without removing it.
          3. The popped message may then be lost to jamming, base drop, or TTL
             expiry. Crucially, a lost message is *consumed* (popped), so a
             "drop" is distinct from "no message" and queues cannot grow under
             pure packet loss.
          4. A surviving message may be spoofed (replaced by noise), then is
             dimension-matched and bandwidth-capped.
        """
        self._ensure_agent(receiver)
        agents_list = list(agents) if agents is not None else (self._agents_cache if self._agents_cache else [receiver])

        zeros = np.zeros((dim,), dtype=float)

        # (1) partition: cannot receive; do not consume queued messages.
        if self._is_partitioned(t, receiver, agents_list):
            return zeros, {"status": "partitioned", "age": 0, "src_payload": None}

        q = self._q[receiver]
        deliverable_idx = [i for i, (dt, _, _, _) in enumerate(q) if dt <= t]
        if not deliverable_idx:
            return zeros, {"status": "none", "age": 0, "src_payload": None}

        # (2) select message: replay peeks, normal delivery pops.
        replayed = False
        if self.cfg.replay_prob > 0 and (self.rng.random() < self.cfg.replay_prob):
            pick = int(self.rng.choice(deliverable_idx))
            dt, ct, src, payload = q[pick]
            replayed = True
        else:
            pick = min(deliverable_idx, key=lambda i: q[i][0])
            dt, ct, src, payload = q.pop(pick)

        # (3-4) channel losses, spoofing, dim, bandwidth.
        return self._finalize_delivery(payload, ct, t, dim, receiver, agents_list, replayed=replayed)

    def _finalize_delivery(self, payload, ct, t, dim, receiver, agents_list, replayed=False):
        """Apply jam/drop/TTL/spoof/dim/bandwidth to one popped message."""
        zeros = np.zeros((dim,), dtype=float)
        src_payload = np.asarray(payload, dtype=float).copy()
        age = int(max(0, t - ct))
        meta: Dict[str, Any] = {"status": "none", "age": age, "src_payload": src_payload}

        if self._is_jammed(t, receiver, agents_list) and (self.rng.random() < self.cfg.jam_drop_prob):
            meta["status"] = "jammed"
            return zeros, meta

        if self.cfg.msg_drop_prob > 0 and (self.rng.random() < self.cfg.msg_drop_prob):
            meta["status"] = "dropped"
            return zeros, meta

        if self.cfg.ttl_steps is not None and age > int(self.cfg.ttl_steps):
            meta["status"] = "stale"
            return zeros, meta

        if self.cfg.spoof_prob > 0 and (self.rng.random() < self.cfg.spoof_prob):
            payload = self.rng.normal(loc=0.0, scale=self.cfg.spoof_scale, size=(dim,)).astype(float)
            meta["status"] = "spoofed"
        else:
            payload = np.asarray(payload, dtype=float)
            meta["status"] = "replayed" if replayed else "delivered"

        if payload.shape[0] < dim:
            out = np.zeros((dim,), dtype=float)
            out[: payload.shape[0]] = payload
            payload = out
        elif payload.shape[0] > dim:
            payload = payload[:dim]

        payload = self._apply_bandwidth_cap(payload)
        return payload.astype(float, copy=False), meta

    def recv_all(
        self,
        receiver: str,
        t: int,
        dim: int,
        agents: Optional[Sequence[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Deliver the freshest deliverable message *from each source*.

        Returns ``{src: (payload, meta)}``. Unlike :meth:`recv` (which returns a
        single message), this drains all currently-deliverable messages, keeping
        only the newest per source, and applies channel impairments independently
        per source. This supports genuine multi-neighbor message passing
        (consensus, formation, DCOP). Partitioned receivers get an empty result.
        """
        self._ensure_agent(receiver)
        agents_list = list(agents) if agents is not None else (self._agents_cache if self._agents_cache else [receiver])

        if self._is_partitioned(t, receiver, agents_list):
            return {}

        q = self._q[receiver]
        # keep latest (max created_t) deliverable message per source
        latest: Dict[str, tuple] = {}
        for (dt, ct, src, payload) in q:
            if dt <= t:
                if (src not in latest) or (ct > latest[src][1]):
                    latest[src] = (dt, ct, src, payload)

        # consume all deliverable messages (keep only not-yet-deliverable)
        self._q[receiver] = [e for e in q if e[0] > t]

        out: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        for src, (dt, ct, _src, payload) in latest.items():
            out[src] = self._finalize_delivery(payload, ct, t, dim, receiver, agents_list, replayed=False)
        return out


# ============================================================
# Core 2: FaultModel (benchmark-agnostic)
# ============================================================
class FaultModel:
    """
    Benchmark-agnostic agent faults:
      - crash-stop / intermittent availability
      - sensor noise on numpy-like observations
      - actuator faults on generic actions (discrete ints, numpy actions, or
        a provided action sampler)
      - byzantine action overriding (random action)

    Crash *onset* is sampled at most once per (agent, timestep): both
    ``transform_observation`` and ``transform_action`` may be called within the
    same step, but only the first call for a given ``t`` samples a new crash.
    This keeps the effective crash probability equal to ``cfg.crash_prob`` and
    keeps the RNG stream reproducible regardless of call order.
    """

    def __init__(self, cfg: NetworkFaultConfig):
        self.cfg = cfg
        self.rng = _spawn_rng(cfg.seed, child=1)
        self._crash_until: Dict[str, int] = {}
        self._last_action: Dict[str, Any] = {}
        self._last_crash_sample_t: Dict[str, int] = {}

    def reset(self, agents: Optional[Sequence[str]] = None, seed: Optional[int] = None):
        base_seed = self.cfg.seed if seed is None else seed
        self.rng = _spawn_rng(base_seed, child=1)
        self._crash_until.clear()
        self._last_action.clear()
        self._last_crash_sample_t.clear()
        if agents is not None:
            for a in agents:
                self._crash_until[a] = -1

    def _is_crashed(self, agent: str, t: int) -> bool:
        return t < self._crash_until.get(agent, -1)

    def begin_step(self, agents: Sequence[str], t: int) -> None:
        """Sample crash onset once for every agent at the start of step ``t``.

        Convenience for harnesses (e.g. the synthetic envs) that drive crash
        state directly rather than through transform_observation/action."""
        for a in agents:
            self._maybe_start_crash(a, t)

    def is_crashed(self, agent: str, t: int) -> bool:
        """Public query: is ``agent`` currently in a crashed state at time ``t``?"""
        return self._is_crashed(agent, t)

    def _maybe_start_crash(self, agent: str, t: int):
        # Sample crash onset at most once per (agent, t).
        if self._last_crash_sample_t.get(agent) == t:
            return
        self._last_crash_sample_t[agent] = t

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

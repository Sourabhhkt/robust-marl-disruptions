# contrib/adapter_experimental.py
#
# EXPERIMENTAL / UNVALIDATED adapters.
# ------------------------------------
# These benchmark adapters are provided as integration *scaffolding* only. They
# were NOT exercised in the validated experiments and are not covered by
# the test suite. They illustrate how the universal wrapper could be connected
# to additional benchmark families (SMAC, Flatland, Overcooked, MovingAI MAPF,
# RoboCup soccer server, FRODO DCOP). Validate before relying on them.
#
# The validated, experiment-backed adapters live in ``adapter.py``.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import re

from adapter import (
    UniversalBenchmarkAdapter,
    inject_msg_into_obs,
    split_action_augmented,
    default_agents,
)


# ============================================================
# SMAC / SMACv2 (obs-as-list world; pair with SMACDictShim)
# ============================================================
@dataclass
class SMACAdapter(UniversalBenchmarkAdapter):
    msg_dim: int
    obs_inject_mode: str = "concat"
    obs_msg_key: str = "msg"
    agent_prefix: str = "agent_"

    def agents(self, env) -> Sequence[str]:
        n = int(getattr(env, "n_agents", 0) or 0)
        if n <= 0:
            n = len(env.get_obs())
        return default_agents(n, self.agent_prefix)

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        return int(self.msg_dim)

    def inject_incoming_message(self, env, agent, obs, msg_in):
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)


class SMACDictShim:
    """Presents SMAC's list-based API as a dict-based env."""

    def __init__(self, smac_env):
        self.env = smac_env
        self.agent_ids = default_agents(self.env.n_agents)

    def reset(self, *args, **kwargs) -> Dict[str, Any]:
        out = self.env.reset(*args, **kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        return {aid: obs[i] for i, aid in enumerate(self.agent_ids)}

    def step(self, action_dict: Dict[str, Any]):
        actions_list = [action_dict[aid] for aid in self.agent_ids]
        out = self.env.step(actions_list)
        reward = out[0]
        terminated = out[1]
        info = out[2] if len(out) > 2 else {}
        obs_list = self.env.get_obs()
        obs = {aid: obs_list[i] for i, aid in enumerate(self.agent_ids)}
        rew = {aid: float(reward) for aid in self.agent_ids}
        done = {aid: bool(terminated) for aid in self.agent_ids}
        infos = {aid: dict(info) for aid in self.agent_ids}
        return obs, rew, done, infos


# ============================================================
# Flatland (RailEnv is already dict-based)
# ============================================================
@dataclass
class FlatlandAdapter(UniversalBenchmarkAdapter):
    msg_dim: int
    obs_inject_mode: str = "dict"
    obs_msg_key: str = "msg"

    def agents(self, env) -> Sequence[str]:
        if hasattr(env, "get_num_agents"):
            n = int(env.get_num_agents())
            return [str(i) for i in range(n)]
        raise ValueError("FlatlandAdapter.agents: env needs get_num_agents()")

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        return int(self.msg_dim)

    def inject_incoming_message(self, env, agent, obs, msg_in):
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)


# ============================================================
# Overcooked-AI (joint-action env; pair with OvercookedDictShim)
# ============================================================
@dataclass
class OvercookedAdapter(UniversalBenchmarkAdapter):
    msg_dim: int
    obs_fn: Callable[[Any, Any], Tuple[np.ndarray, np.ndarray]]
    obs_inject_mode: str = "concat"
    agent_ids: Tuple[str, str] = ("player_0", "player_1")

    def agents(self, env) -> Sequence[str]:
        return list(self.agent_ids)

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        return int(self.msg_dim)

    def inject_incoming_message(self, env, agent, obs, msg_in):
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode)

    def to_joint_action(self, action_dict):
        return (action_dict[self.agent_ids[0]], action_dict[self.agent_ids[1]])

    def from_state(self, env, state):
        o0, o1 = self.obs_fn(env, state)
        return {self.agent_ids[0]: o0, self.agent_ids[1]: o1}


class OvercookedDictShim:
    def __init__(self, oc_env, adapter: OvercookedAdapter):
        self.env = oc_env
        self.adapter = adapter
        self._state = None

    def reset(self, *args, **kwargs):
        self._state = self.env.reset(*args, **kwargs)
        return self.adapter.from_state(self.env, self._state)

    def step(self, action_dict):
        joint = self.adapter.to_joint_action(action_dict)
        out = self.env.step(joint)
        next_state, reward, done = out[0], out[1], out[2]
        info = out[3] if len(out) > 3 else {}
        self._state = next_state
        obs = self.adapter.from_state(self.env, next_state)
        rew = {aid: float(reward) for aid in self.adapter.agent_ids}
        dones = {aid: bool(done) for aid in self.adapter.agent_ids}
        infos = {aid: dict(info) for aid in self.adapter.agent_ids}
        return obs, rew, dones, infos


# ============================================================
# MovingAI MAPF (dataset parsers + harness adapter)
# ============================================================
def parse_movingai_map(path: str) -> np.ndarray:
    """MovingAI octile map -> grid (0=free, 1=blocked)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    assert lines[0].startswith("type"), "Invalid map file: missing type"
    h = int(lines[1].split()[-1])
    w = int(lines[2].split()[-1])
    assert lines[3].strip() == "map", "Invalid map file: missing 'map' line"
    grid_lines = lines[4:4 + h]
    grid = np.zeros((h, w), dtype=np.uint8)
    blocked = {"@", "T"}
    for r in range(h):
        row = grid_lines[r]
        for c, ch in enumerate(row):
            grid[r, c] = 1 if ch in blocked else 0
    return grid


def parse_movingai_scen(path: str, k: int):
    """Parse first k start/goal pairs from a MovingAI .scen file."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    start_idx = 1 if lines[0].lower().startswith("version") else 0
    for ln in lines[start_idx:start_idx + k]:
        parts = ln.split()
        sx = int(parts[-5]); sy = int(parts[-4]); gx = int(parts[-3]); gy = int(parts[-2])
        tasks.append(((sx, sy), (gx, gy)))
    return tasks


@dataclass
class MAPFHarnessAdapter(UniversalBenchmarkAdapter):
    msg_dim: int
    obs_inject_mode: str = "dict"
    obs_msg_key: str = "msg"

    def agents(self, env) -> Sequence[str]:
        return list(env.agents)

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        return int(self.msg_dim)

    def inject_incoming_message(self, env, agent, obs, msg_in):
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)


# ============================================================
# RoboCup soccer server aural-comm UDP proxy
# ============================================================
_HEAR_RE = re.compile(r"^\(hear\s+(\d+)\s+([^\s]+)\s+\"(.*)\"\)$")


@dataclass
class RoboCupAuralProxyConfig:
    server_addr: Tuple[str, int]
    bind_addr: Tuple[str, int]
    client_addr: Optional[Tuple[str, int]] = None
    msg_dim: int = 32
    encode: Callable[[str, int], np.ndarray] = lambda s, d: np.frombuffer(s.encode("utf-8")[:d], dtype=np.uint8).astype(float)
    decode: Callable[[np.ndarray], str] = lambda a: bytes(np.asarray(a, dtype=np.uint8)).decode("utf-8", errors="ignore").rstrip("\x00")


class RoboCupUDPProxy:
    """Intercepts say/hear between a RoboCup client and server to apply channel faults."""

    def __init__(self, cfg: RoboCupAuralProxyConfig, channel, *, agent_id: str, agents: Sequence[str]):
        import socket
        self.cfg = cfg
        self.channel = channel
        self.agent_id = agent_id
        self.agents = list(agents)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(cfg.bind_addr)
        self.sock.setblocking(False)

    def _is_say(self, msg: str) -> Optional[str]:
        if msg.startswith("(say"):
            m = re.search(r"\"(.*)\"", msg)
            return m.group(1) if m else ""
        return None

    def loop_once(self, t: int):
        import select
        r, _, _ = select.select([self.sock], [], [], 0.0)
        if not r:
            return
        data, addr = self.sock.recvfrom(65535)
        text = data.decode("utf-8", errors="ignore").strip()
        if self.cfg.client_addr is None and addr != self.cfg.server_addr:
            self.cfg.client_addr = addr

        if addr == self.cfg.client_addr:
            say_payload = self._is_say(text)
            if say_payload is None:
                self.sock.sendto(data, self.cfg.server_addr)
                return
            vec = self.cfg.encode(say_payload, self.cfg.msg_dim)
            for other in self.agents:
                if other != self.agent_id:
                    self.channel.send(self.agent_id, other, vec, t, agents=self.agents)
            self.sock.sendto(data, self.cfg.server_addr)
            return

        if addr == self.cfg.server_addr:
            m = _HEAR_RE.match(text)
            if not m or self.cfg.client_addr is None:
                self.sock.sendto(data, self.cfg.client_addr)
                return
            # recv now returns (payload, meta)
            msg_in, _meta = self.channel.recv(self.agent_id, t, dim=self.cfg.msg_dim, agents=self.agents)
            new_txt = self.cfg.decode(msg_in)
            time_s, sender_s, _ = m.group(1), m.group(2), m.group(3)
            new_packet = f'(hear {time_s} {sender_s} "{new_txt}")'.encode("utf-8")
            self.sock.sendto(new_packet, self.cfg.client_addr)


# ============================================================
# FRODO DCOP (JPype bridge skeleton)
# ============================================================
@dataclass
class FrodoJPypeHookConfig:
    classpath: Sequence[str]
    msg_dim: int = 64


class FrodoJPypeBridge:
    """Skeleton: install a custom IncomingMsgPolicy that applies channel faults.
    NOTE: the validated DCOP path uses the pure-Python ``DCOPEnv`` in
    ``synth_envs.py`` (max-sum / DSA), not this Java bridge."""

    def __init__(self, cfg: FrodoJPypeHookConfig, channel):
        self.cfg = cfg
        self.channel = channel
        self._started = False

    def start_jvm(self):
        import jpype
        import jpype.imports  # noqa: F401
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=list(self.cfg.classpath))
        self._started = True

    def make_incoming_policy(self, agent_id: str, agents: Sequence[str]):
        if not self._started:
            self.start_jvm()
        import jpype
        from jpype import JImplements, JOverride
        IncomingIface = jpype.JClass("frodo2.communication.IncomingMsgPolicyInterface")

        @JImplements(IncomingIface)
        class Policy:
            def __init__(self):
                self.t = 0

            @JOverride
            def notifyIn(self, msg):
                return msg

        return Policy()

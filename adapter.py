# adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import re
import socket
import select


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
        if hasattr(space, "shape"):
            return np.zeros(space.shape, dtype=np.float32)
        return 0

    def random_action(self, env, agent: str) -> Any:
        return env.action_space(agent).sample()


@dataclass
class ObsSliceCommAdapter(UniversalBenchmarkAdapter):
    agents_list_fn: Callable[[Any], Sequence[str]]
    slice_fn: Callable[[Any], np.ndarray]
    replace_fn: Callable[[Any, np.ndarray], Any]

    def agents(self, env) -> Sequence[str]:
        return list(self.agents_list_fn(env))

    def message_dim(self, env, agent: str, obs: Any = None) -> int:
        if obs is None:
            raise ValueError("ObsSliceCommAdapter.message_dim needs obs")
        return int(self.slice_fn(obs).shape[0])

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
        return { "observation": obs, key: msg_in }

    # concat
    base = coerce_1d_float(obs)
    return np.concatenate([base, msg_in.astype(float, copy=False)], axis=0)


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
# Generic dict-based adapter (PettingZoo Parallel, Flatland, many MARL libs)
# ============================================================

@dataclass
class DictMAAdapter:
    """
    Works when env.reset() and env.step() use dicts keyed by agent_id.
    - actions: Dict[agent_id] -> env_action  OR (env_action, msg_out)
    - obs: Dict[agent_id] -> obs_vector / obs_struct

    You decide how to inject msg into obs via obs_inject_mode.
    """
    agents_list_fn: Callable[[Any], Sequence[str]]
    msg_dim: int
    obs_inject_mode: str = "dict"      # "dict" | "concat" | "tuple" | "noop"
    obs_msg_key: str = "msg"

    def agents(self, env) -> Sequence[str]:
        return list(self.agents_list_fn(env))

    def split_action(self, agent: str, action: Any) -> Tuple[Any, Optional[np.ndarray]]:
        return split_action_augmented(action)

    def join_observation(self, agent: str, obs: Any, msg_in: np.ndarray) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)

    def message_dim(self, env, agent: str) -> int:
        return int(self.msg_dim)


# ============================================================
# 1) SMAC / SMACv2 adapter
# ============================================================

@dataclass
class SMACAdapter:
    """
    SMAC/SMACv2 environments commonly expose:
      - n_agents
      - reset() -> (obs, state)  OR obs
      - get_obs() -> List[np.ndarray] (n_agents x obs_dim)
      - step(actions: List[int]) -> reward, terminated, info (varies by wrapper)

    SMACv2 keeps SMAC API.  (See SMACv2 README / paper references.) :contentReference[oaicite:8]{index=8}
    torchrl's wrapper calls env.get_obs(), env.get_state(), env.reset() -> (obs, state). :contentReference[oaicite:9]{index=9}

    This adapter is for the "obs as list" world. You'll typically pair it with a small SMAC wrapper
    that converts list<->dict for the UniversalCommFaultWrapper.
    """
    msg_dim: int
    obs_inject_mode: str = "concat"   # SMAC obs is usually 1D numeric
    obs_msg_key: str = "msg"
    agent_prefix: str = "agent_"

    def agents(self, env) -> Sequence[str]:
        n = int(getattr(env, "n_agents", None) or getattr(env, "n_agents", 0) or 0)
        if n <= 0:
            # fallback: infer from get_obs()
            obs = env.get_obs()
            n = len(obs)
        return default_agents(n, self.agent_prefix)

    def split_action(self, agent: str, action: Any) -> Tuple[Any, Optional[np.ndarray]]:
        return split_action_augmented(action)

    def join_observation(self, agent: str, obs: Any, msg_in: np.ndarray) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)

    def message_dim(self, env, agent: str) -> int:
        return int(self.msg_dim)


class SMACDictShim:
    """
    Thin shim that presents SMAC as a dict-based env:
      reset()  -> obs_dict
      step(a_dict) -> obs_dict, reward_dict, done_dict, info_dict  (minimal)

    NOTE: SMAC's native step signature varies across wrappers; adjust in one place if needed.
    """
    def __init__(self, smac_env):
        self.env = smac_env
        self.agent_ids = default_agents(self.env.n_agents)

    def reset(self, *args, **kwargs) -> Dict[str, Any]:
        out = self.env.reset(*args, **kwargs)
        # could be (obs, state) or obs
        obs = out[0] if isinstance(out, tuple) else out
        return {aid: obs[i] for i, aid in enumerate(self.agent_ids)}

    def step(self, action_dict: Dict[str, Any]):
        actions_list = [action_dict[aid] for aid in self.agent_ids]
        out = self.env.step(actions_list)

        # Common SMAC variants:
        #   reward, terminated, info
        #   reward, terminated, info, ...
        reward = out[0]
        terminated = out[1]
        info = out[2] if len(out) > 2 else {}

        obs_list = self.env.get_obs()
        obs = {aid: obs_list[i] for i, aid in enumerate(self.agent_ids)}
        rew = {aid: float(reward) for aid in self.agent_ids}  # team reward broadcast
        done = {aid: bool(terminated) for aid in self.agent_ids}
        infos = {aid: dict(info) for aid in self.agent_ids}
        return obs, rew, done, infos


# ============================================================
# 2) MAgent adapter (use MAgent2 / PettingZoo magent)
# ============================================================

# In practice, treat MAgent2 as PettingZoo Parallel, so DictMAAdapter works.
# MAgent2 is a Python gridworld multi-agent library. :contentReference[oaicite:10]{index=10}
# PettingZoo Parallel API describes dict obs/actions. :contentReference[oaicite:11]{index=11}


# ============================================================
# 3) Flatland adapter
# ============================================================

@dataclass
class FlatlandAdapter:
    """
    Flatland's RailEnv is already dict-based: step(action_dict) returns dicts. :contentReference[oaicite:12]{index=12}
    Observations can be tree/structured depending on ObservationBuilder; safest inject mode is "dict" or "tuple".
    """
    msg_dim: int
    obs_inject_mode: str = "dict"
    obs_msg_key: str = "msg"

    def agents(self, env) -> Sequence[str]:
        # Flatland agents are usually handles 0..n-1; use env.get_num_agents() when available
        if hasattr(env, "get_num_agents"):
            n = int(env.get_num_agents())
            return [str(i) for i in range(n)]
        # fallback: infer from last observations
        raise ValueError("FlatlandAdapter.agents: provide env with get_num_agents() or wrap to supply agent ids")

    def split_action(self, agent: str, action: Any) -> Tuple[Any, Optional[np.ndarray]]:
        return split_action_augmented(action)

    def join_observation(self, agent: str, obs: Any, msg_in: np.ndarray) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)

    def message_dim(self, env, agent: str) -> int:
        return int(self.msg_dim)


# ============================================================
# 4) Overcooked-AI adapter
# ============================================================

@dataclass
class OvercookedAdapter:
    """
    Overcooked-AI (HumanCompatibleAI) commonly uses joint-action stepping. :contentReference[oaicite:13]{index=13}
    This adapter exposes it as two agents: "player_0", "player_1".

    Assumptions (typical):
      - env.reset() returns something representing state/obs
      - env.step((a0, a1)) returns next_state, reward, done, info (varies by wrapper)
      - you can obtain per-player observations via a provided featurizer / state encoding
    Because Overcooked codebases differ, you MUST supply:
      obs_fn(env, state) -> (obs0_vec, obs1_vec)  (each 1D np array)
    """
    msg_dim: int
    obs_fn: Callable[[Any, Any], Tuple[np.ndarray, np.ndarray]]
    obs_inject_mode: str = "concat"
    agent_ids: Tuple[str, str] = ("player_0", "player_1")

    def agents(self, env) -> Sequence[str]:
        return list(self.agent_ids)

    def split_action(self, agent: str, action: Any) -> Tuple[Any, Optional[np.ndarray]]:
        return split_action_augmented(action)

    def message_dim(self, env, agent: str) -> int:
        return int(self.msg_dim)

    def join_observation(self, agent: str, obs: Any, msg_in: np.ndarray) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode)

    def to_joint_action(self, action_dict: Dict[str, Any]) -> Tuple[Any, Any]:
        return (action_dict[self.agent_ids[0]], action_dict[self.agent_ids[1]])

    def from_state(self, env, state) -> Dict[str, np.ndarray]:
        o0, o1 = self.obs_fn(env, state)
        return {self.agent_ids[0]: o0, self.agent_ids[1]: o1}


class OvercookedDictShim:
    """
    Presents an Overcooked env with joint actions as dict-based env for UniversalCommFaultWrapper.

    You supply an adapter with obs_fn() that extracts per-agent observations from the env's state.
    """
    def __init__(self, oc_env, adapter: OvercookedAdapter):
        self.env = oc_env
        self.adapter = adapter
        self._state = None

    def reset(self, *args, **kwargs) -> Dict[str, Any]:
        self._state = self.env.reset(*args, **kwargs)
        return self.adapter.from_state(self.env, self._state)

    def step(self, action_dict: Dict[str, Any]):
        joint = self.adapter.to_joint_action(action_dict)
        out = self.env.step(joint)
        # typical: next_state, reward, done, info
        next_state, reward, done, info = out[0], out[1], out[2], out[3] if len(out) > 3 else {}
        self._state = next_state
        obs = self.adapter.from_state(self.env, next_state)
        rew = {aid: float(reward) for aid in self.adapter.agent_ids}     # team reward broadcast
        dones = {aid: bool(done) for aid in self.adapter.agent_ids}
        infos = {aid: dict(info) for aid in self.adapter.agent_ids}
        return obs, rew, dones, infos



class Grid2OpParallelEnv:
    def __init__(self, env_name="l2rpn_case14_sandbox", max_steps=100, **kwargs):
        import grid2op

        self.env = grid2op.make(env_name, **kwargs)
        self.agent_ids = ["grid_operator"]
        self._active_agents = list(self.agent_ids)
        self.max_steps = int(max_steps)
        self._step_count = 0

    @property
    def agents(self):
        return self._active_agents

    @property
    def possible_agents(self):
        return self.agent_ids

    def action_space(self, agent):
        return self.env.action_space

    def observation_space(self, agent):
        return None

    def _obs_to_vec(self, obs):
        try:
            return np.asarray(obs.to_vect(), dtype=np.float32).reshape(-1)
        except Exception:
            return np.asarray(obs, dtype=np.float32).reshape(-1)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._active_agents = list(self.agent_ids)
        try:
            obs = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs = self.env.reset()
        return {"grid_operator": self._obs_to_vec(obs)}, {"grid_operator": {}}

    def step(self, actions):
        act = actions.get("grid_operator", self.env.action_space({}))
        out = self.env.step(act)

        if len(out) == 4:
            obs, reward, done, info = out
            truncated = False
        else:
            obs, reward, done, truncated, info = out

        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True

        if done or truncated:
            self._active_agents = []
        else:
            self._active_agents = list(self.agent_ids)

        return (
            {"grid_operator": self._obs_to_vec(obs)},
            {"grid_operator": float(reward)},
            {"grid_operator": bool(done)},
            {"grid_operator": bool(truncated)},
            {"grid_operator": dict(info) if isinstance(info, dict) else {}},
        )

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


@dataclass
class Grid2OpCommAdapter(ObsSliceCommAdapter):
    def noop_action(self, env, agent: str) -> Any:
        return env.action_space(agent)({})

    def random_action(self, env, agent: str) -> Any:
        space = env.action_space(agent)
        try:
            return space.sample()
        except Exception:
            # safest fallback if sample is unavailable/brittle
            return space({})

# ============================================================
# 5) Hanabi adapter (recommended: PettingZoo Hanabi)
# ============================================================

# PettingZoo Hanabi is already multi-agent with dict observations (incl action_mask). :contentReference[oaicite:14]{index=14}
# So DictMAAdapter works directly.

# If you insist on DeepMind hanabi-learning-environment (rl_env.py), it's not naturally multi-agent;
# it's a single environment returning a dict observation with current-player perspective.
# In that case, "comm loss" is best modeled by masking parts of observation updates caused by hint actions.
# That’s invasive and version-specific, so I’m not giving a misleading “generic” adapter here.


# ============================================================
# 6) MovingAI MAPF adapter (dataset -> you own the harness)
# ============================================================

def parse_movingai_map(path: str) -> np.ndarray:
    """
    MovingAI map format (octile) is documented: type/height/width/map + ASCII grid. :contentReference[oaicite:15]{index=15}
    Returns grid with 0=free, 1=blocked.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    assert lines[0].startswith("type"), "Invalid map file: missing type"
    h = int(lines[1].split()[-1])
    w = int(lines[2].split()[-1])
    assert lines[3].strip() == "map", "Invalid map file: missing 'map' line"

    grid_lines = lines[4:4+h]
    grid = np.zeros((h, w), dtype=np.uint8)
    # '.' and 'G' etc are traversable; '@' and 'T' etc are blocked in MovingAI conventions
    blocked = set(["@", "T"])
    for r in range(h):
        row = grid_lines[r]
        for c, ch in enumerate(row):
            grid[r, c] = 1 if ch in blocked else 0
    return grid


def parse_movingai_scen(path: str, k: int) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Typical .scen format provides start/goal pairs for many instances.
    The MAPF benchmark page links these datasets. :contentReference[oaicite:16]{index=16}
    We'll parse first k tasks: ((sx,sy),(gx,gy)).
    """
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # first line often: "version 1"
    start_idx = 1 if lines[0].lower().startswith("version") else 0

    for ln in lines[start_idx:start_idx+k]:
        # common columns: bucket map width height sx sy gx gy dist
        parts = ln.split()
        # robust: take last 5 numeric tokens -> width height sx sy gx gy dist varies
        # Most common: ... sx sy gx gy ...
        sx = int(parts[-5]); sy = int(parts[-4]); gx = int(parts[-3]); gy = int(parts[-2])
        tasks.append(((sx, sy), (gx, gy)))
    return tasks


@dataclass
class MAPFHarnessAdapter:
    """
    You build a Python MAPF harness with dict obs/actions; this adapter is the same pattern as DictMAAdapter.
    The important part is: MAPF is not an env with comm, so you define msg_dim + injection.
    """
    msg_dim: int
    obs_inject_mode: str = "dict"
    obs_msg_key: str = "msg"

    def agents(self, env) -> Sequence[str]:
        return list(env.agents)  # expect harness exposes env.agents

    def split_action(self, agent: str, action: Any) -> Tuple[Any, Optional[np.ndarray]]:
        return split_action_augmented(action)

    def join_observation(self, agent: str, obs: Any, msg_in: np.ndarray) -> Any:
        return inject_msg_into_obs(obs, msg_in, mode=self.obs_inject_mode, key=self.obs_msg_key)

    def message_dim(self, env, agent: str) -> int:
        return int(self.msg_dim)


# ============================================================
# 7) RoboCup Soccer Server / Sim adapter (Python UDP proxy skeleton)
# ============================================================

_HEAR_RE = re.compile(r"^\(hear\s+(\d+)\s+([^\s]+)\s+\"(.*)\"\)$")

@dataclass
class RoboCupAuralProxyConfig:
    """
    RoboCup Soccer Server uses text protocol over UDP; comm is via say/hear.
    Aural sensor model: (hear Time Sender "Message"). :contentReference[oaicite:17]{index=17}

    This proxy sits between *one client* and the server:
      client <-> proxy <-> server
    and lets you apply NetworkChannel impairments to say/hear.
    """
    server_addr: Tuple[str, int]
    bind_addr: Tuple[str, int]
    client_addr: Optional[Tuple[str, int]] = None  # learned dynamically if None

    msg_dim: int = 32
    encode: Callable[[str, int], np.ndarray] = lambda s, d: np.frombuffer(s.encode("utf-8")[:d], dtype=np.uint8).astype(float)
    decode: Callable[[np.ndarray], str] = lambda a: bytes(np.asarray(a, dtype=np.uint8)).decode("utf-8", errors="ignore").rstrip("\x00")


class RoboCupUDPProxy:
    """
    Minimal UDP proxy to intercept:
      - outgoing "(say \"...\")" commands from client
      - incoming "(hear ...)" messages from server

    Use NetworkChannel to drop/delay/corrupt messages in Python, even though the server delivers immediately.
    """
    def __init__(self, cfg: RoboCupAuralProxyConfig, channel, *, agent_id: str, agents: Sequence[str]):
        self.cfg = cfg
        self.channel = channel
        self.agent_id = agent_id
        self.agents = list(agents)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(cfg.bind_addr)
        self.sock.setblocking(False)

    def _is_say(self, msg: str) -> Optional[str]:
        # common: (say "Message")
        if msg.startswith("(say"):
            # naive parse:
            m = re.search(r"\"(.*)\"", msg)
            return m.group(1) if m else ""
        return None

    def loop_once(self, t: int):
        """
        Call in a scheduler; moves packets between client and server with optional comm faults.
        """
        r, _, _ = select.select([self.sock], [], [], 0.0)
        if not r:
            return

        data, addr = self.sock.recvfrom(65535)
        text = data.decode("utf-8", errors="ignore").strip()

        # learn client addr
        if self.cfg.client_addr is None and addr != self.cfg.server_addr:
            self.cfg.client_addr = addr

        # direction: client->proxy (addr==client) OR server->proxy (addr==server)
        if addr == self.cfg.client_addr:
            say_payload = self._is_say(text)
            if say_payload is None:
                # not a say: forward raw to server
                self.sock.sendto(data, self.cfg.server_addr)
                return

            # encode & send to others via channel (you define recipients; here: broadcast)
            vec = self.cfg.encode(say_payload, self.cfg.msg_dim)
            for other in self.agents:
                if other != self.agent_id:
                    self.channel.send(self.agent_id, other, vec, t, agents=self.agents)

            # still forward original say to server (optional). If you want to fully own comm,
            # you can choose NOT to forward and instead synthesize hear packets yourself.
            self.sock.sendto(data, self.cfg.server_addr)
            return

        if addr == self.cfg.server_addr:
            # intercept hear packets; if it's a hear message, optionally replace with impaired version
            m = _HEAR_RE.match(text)
            if not m or self.cfg.client_addr is None:
                self.sock.sendto(data, self.cfg.client_addr)
                return

            # deliver our (possibly impaired) message to THIS agent from channel
            msg_in = self.channel.recv(self.agent_id, t, dim=self.cfg.msg_dim, agents=self.agents)
            new_txt = self.cfg.decode(msg_in)

            # rebuild hear packet (keep time/sender but replace message)
            time_s, sender_s, _ = m.group(1), m.group(2), m.group(3)
            new_packet = f'(hear {time_s} {sender_s} "{new_txt}")'.encode("utf-8")
            self.sock.sendto(new_packet, self.cfg.client_addr)


# ============================================================
# 8) FRODO DCOP adapter (JPype bridge skeleton)
# ============================================================

@dataclass
class FrodoJPypeHookConfig:
    """
    FRODO is primarily Java; experiments run via their framework. :contentReference[oaicite:18]{index=18}
    They define incoming message listener/policy interfaces. :contentReference[oaicite:19]{index=19}
    In Python, you can bridge via JPype to install a custom IncomingMsgPolicyInterface that
    applies your NetworkChannel faults before delivering messages.

    This is a skeleton: wiring depends on which FRODO solver + queue classes you instantiate.
    """
    classpath: Sequence[str]
    msg_dim: int = 64


class FrodoJPypeBridge:
    """
    Skeleton pattern (not runnable without your exact FRODO experiment setup).
    """
    def __init__(self, cfg: FrodoJPypeHookConfig, channel):
        self.cfg = cfg
        self.channel = channel
        self._started = False

    def start_jvm(self):
        import jpype
        import jpype.imports
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=list(self.cfg.classpath))
        self._started = True

    def make_incoming_policy(self, agent_id: str, agents: Sequence[str]):
        """
        Returns a Java object implementing IncomingMsgPolicyInterface that:
          - onMessage(msg): encodes msg -> vector, applies channel, decodes -> msg', returns msg'
        """
        if not self._started:
            self.start_jvm()

        import jpype
        from jpype import JImplements, JOverride

        # You must import the right interface from FRODO jar.
        # Docs show frodo2.communication.IncomingMsgPolicyInterface exists. :contentReference[oaicite:20]{index=20}
        IncomingIface = jpype.JClass("frodo2.communication.IncomingMsgPolicyInterface")

        @JImplements(IncomingIface)
        class Policy:
            def __init__(self):
                self.t = 0

            @JOverride
            def notifyIn(self, msg):  # method name may differ; inspect FRODO interface in your jar
                # TODO: implement encoding/decoding for your message class
                # Example idea:
                #   payload_str = msg.toString()
                #   vec = encode(payload_str)
                #   vec2 = channel.recv(agent_id,...)
                #   msg2 = reconstruct msg
                return msg

        return Policy()

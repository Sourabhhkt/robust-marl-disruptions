from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np

from core import NetworkChannel, FaultModel, NetworkFaultConfig


class InstrumentedUniversalCommFaultWrapper:
    """
    Universal communication-fault wrapper that works with:
      1) explicit-message benchmarks
      2) implicit observation-slice communication benchmarks

    It delegates benchmark-specific logic to the adapter.

    Expected env interface:
      - reset(seed=None, options=None) -> (obs_dict, infos)
      - step(action_dict) -> (obs_dict, rewards, terms, truncs, infos)
      - agents
      - possible_agents
      - action_space(agent)
      - observation_space(agent)

    Expected adapter interface:
      - split_action(agent, action) -> (env_action, explicit_msg_out)
      - message_dim(env, agent, obs=None) -> int
      - extract_outgoing_message(env, agent, obs, env_action, explicit_msg_out) -> np.ndarray
      - inject_incoming_message(env, agent, obs, msg_in) -> obs
      - noop_action(env, agent)
      - random_action(env, agent)
    """

    def __init__(
        self,
        env,
        adapter,
        cfg: NetworkFaultConfig,
        routing_fn: Optional[Callable[[str, Sequence[str]], List[str]]] = None,
    ):
        self.env = env
        self.adapter = adapter
        self.cfg = cfg

        self.channel = NetworkChannel(cfg)
        self.fault_model = FaultModel(cfg)

        self.routing_fn = routing_fn or self._default_routing
        self.t = 0

        self._last_raw_obs: Dict[str, Any] = {}
        self.reset_logs()

    # ---------------------------------------------------------
    # passthroughs
    # ---------------------------------------------------------

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None

    # ---------------------------------------------------------
    # logging
    # ---------------------------------------------------------

    def reset_logs(self):
        self.logs = {
            # communication quality on receive side
            "mse_sum": 0.0,
            "mse_n": 0,
            "zero_sum": 0,
            "zero_n": 0,
            "age_sum": 0.0,
            "age_n": 0,

            # channel event counts
            "sent_msgs": 0,
            "recv_attempts": 0,
            "delivered_real_msgs": 0,

            # step-level bookkeeping
            "steps": 0,
        }

    # ---------------------------------------------------------
    # routing
    # ---------------------------------------------------------

    def _default_routing(self, src: str, agents: Sequence[str]) -> List[str]:
        """
        Default routing:
        - multi-agent: broadcast to all other active agents
        - single-agent: self-loop
        """
        agents = list(agents)
        if len(agents) <= 1:
            return [src]
        return [a for a in agents if a != src]

    # ---------------------------------------------------------
    # helper utilities
    # ---------------------------------------------------------

    @staticmethod
    def _to_1d_float_array(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return arr

    def _safe_message_dim(self, agent: str, obs: Any = None) -> int:
        try:
            d = int(self.adapter.message_dim(self.env, agent, obs))
            return max(0, d)
        except Exception:
            return 0

    def _record_comm_distortion(
        self,
        raw_msg: Optional[np.ndarray],
        delivered_msg: Optional[np.ndarray],
    ) -> None:
        if raw_msg is None or delivered_msg is None:
            return

        raw = self._to_1d_float_array(raw_msg)
        rec = self._to_1d_float_array(delivered_msg)

        if raw.shape != rec.shape:
            m = min(raw.shape[0], rec.shape[0])
            raw = raw[:m]
            rec = rec[:m]

        if raw.size == 0:
            return

        self.logs["mse_sum"] += float(np.mean((raw - rec) ** 2))
        self.logs["mse_n"] += 1
        self.logs["zero_sum"] += int(np.sum(rec == 0.0))
        self.logs["zero_n"] += int(rec.size)

        if np.any(rec != 0.0):
            self.logs["delivered_real_msgs"] += 1

    def _extract_raw_outgoing_message(
        self,
        agent: str,
        raw_obs: Any,
        env_action: Any,
        explicit_msg_out: Optional[np.ndarray],
    ) -> np.ndarray:
        msg = self.adapter.extract_outgoing_message(
            self.env,
            agent,
            raw_obs,
            env_action,
            explicit_msg_out,
        )
        msg = self._to_1d_float_array(msg)
        return msg

    # ---------------------------------------------------------
    # observation transform
    # ---------------------------------------------------------

    def _transform_observations(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        active_agents = list(obs_dict.keys())

        for agent, obs in obs_dict.items():
            # first apply non-comm observation faults (sensor noise, crash masking, etc.)
            obs_faulted = self.fault_model.transform_observation(agent, obs, self.t)

            msg_dim = self._safe_message_dim(agent, obs_faulted)
            if msg_dim <= 0:
                out[agent] = obs_faulted
                continue

            self.logs["recv_attempts"] += 1
            msg_in = self.channel.recv(
                receiver=agent,
                t=self.t,
                dim=msg_dim,
                agents=active_agents,
            )
            msg_in = self._to_1d_float_array(msg_in)

            # for instrumentation: compare delivered msg against the adapter's raw slice
            try:
                raw_obs = self._last_raw_obs.get(agent, obs)
                raw_msg = self.adapter.extract_outgoing_message(
                    self.env,
                    agent,
                    raw_obs,
                    env_action=None,
                    explicit_msg_out=None,
                )
                raw_msg = self._to_1d_float_array(raw_msg)
            except Exception:
                raw_msg = None

            self._record_comm_distortion(raw_msg, msg_in)

            # approximate age tracking: if a non-zero payload arrives, use base latency as proxy
            if msg_in.size > 0 and np.any(msg_in != 0.0):
                approx_age = max(0, int(getattr(self.cfg, "base_latency_steps", 0)))
                self.logs["age_sum"] += float(approx_age)
                self.logs["age_n"] += 1

            obs_with_msg = self.adapter.inject_incoming_message(
                self.env,
                agent,
                obs_faulted,
                msg_in,
            )
            out[agent] = obs_with_msg

        return out

    # ---------------------------------------------------------
    # action transform + send side
    # ---------------------------------------------------------

    def _transform_actions(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        env_actions: Dict[str, Any] = {}
        active_agents = list(action_dict.keys())

        for agent, action in action_dict.items():
            env_action, explicit_msg_out = self.adapter.split_action(agent, action)

            env_action = self.fault_model.transform_action(
                agent,
                env_action,
                self.t,
                noop_action_fn=lambda agent=agent: self.adapter.noop_action(self.env, agent),
                random_action_fn=lambda agent=agent: self.adapter.random_action(self.env, agent),
            )

            raw_obs = self._last_raw_obs.get(agent, None)
            try:
                msg_out = self._extract_raw_outgoing_message(
                    agent=agent,
                    raw_obs=raw_obs,
                    env_action=env_action,
                    explicit_msg_out=explicit_msg_out,
                )
            except Exception:
                msg_out = np.zeros((0,), dtype=float)

            if msg_out.size > 0:
                dsts = list(self.routing_fn(agent, active_agents))
                for dst in dsts:
                    self.channel.send(
                        src=agent,
                        dst=dst,
                        payload=msg_out,
                        t=self.t,
                        agents=active_agents,
                        allow_byzantine_corrupt=True,
                    )
                    self.logs["sent_msgs"] += 1

            env_actions[agent] = env_action

        return env_actions

    # ---------------------------------------------------------
    # main API
    # ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.t = 0
        self.reset_logs()

        out = self.env.reset(seed=seed, options=options)

        if isinstance(out, tuple) and len(out) == 2:
            obs, infos = out
        else:
            obs = out
            infos = {a: {} for a in getattr(self.env, "agents", [])}

        active_agents = list(obs.keys())

        self.channel.reset(agents=active_agents, seed=seed)
        self.fault_model.reset(agents=active_agents, seed=seed)

        self._last_raw_obs = {a: obs[a] for a in obs}
        obs = self._transform_observations(obs)
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        env_actions = self._transform_actions(actions)

        obs, rewards, terms, truncs, infos = self.env.step(env_actions)

        self.t += 1
        self.logs["steps"] += 1

        self._last_raw_obs = {a: obs[a] for a in obs}
        obs = self._transform_observations(obs)

        return obs, rewards, terms, truncs, infos


# class UniversalCommFaultWrapper:
#     def __init__(self, env, adapter, cfg, routing_fn=None):
#         self.env = env
#         self.adapter = adapter
#         self.cfg = cfg
#         self.channel = NetworkChannel(cfg)
#         self.fault_model = FaultModel(cfg)
#         self.routing_fn = routing_fn or self._broadcast_routing
#         self.t = 0
#         self._last_raw_obs = {}

#     @property
#     def agents(self):
#         return self.env.agents

#     @property
#     def possible_agents(self):
#         return self.env.possible_agents

#     def action_space(self, agent):
#         return self.env.action_space(agent)

#     def observation_space(self, agent):
#         return self.env.observation_space(agent)

#     def close(self):
#         return self.env.close()

#     def _broadcast_routing(self, src, agents):
#         return [a for a in agents if a != src]

#     def _transform_obs(self, obs_dict):
#         out = {}
#         agents_now = list(obs_dict.keys())

#         for a, obs in obs_dict.items():
#             obs2 = self.fault_model.transform_observation(a, obs, self.t)

#             msg_dim = self.adapter.message_dim(self.env, a, obs2)
#             msg_in = self.channel.recv(a, self.t, msg_dim, agents=agents_now)

#             obs3 = self.adapter.inject_incoming_message(self.env, a, obs2, msg_in)
#             out[a] = obs3

#         return out

#     def _transform_actions(self, action_dict):
#         env_actions = {}
#         agents_now = list(action_dict.keys())

#         for a, action in action_dict.items():
#             env_action, explicit_msg_out = self.adapter.split_action(a, action)

#             env_action = self.fault_model.transform_action(
#                 a,
#                 env_action,
#                 self.t,
#                 noop_action_fn=lambda a=a: self.adapter.noop_action(self.env, a),
#                 random_action_fn=lambda a=a: self.adapter.random_action(self.env, a),
#             )

#             raw_obs = self._last_raw_obs.get(a, None)
#             msg_out = self.adapter.extract_outgoing_message(
#                 self.env, a, raw_obs, env_action, explicit_msg_out
#             )

#             for dst in self.routing_fn(a, agents_now):
#                 self.channel.send(
#                     src=a,
#                     dst=dst,
#                     payload=msg_out,
#                     t=self.t,
#                     agents=agents_now,
#                     allow_byzantine_corrupt=True,
#                 )

#             env_actions[a] = env_action

#         return env_actions

#     def reset(self, seed=None, options=None):
#         self.t = 0
#         obs, infos = self.env.reset(seed=seed, options=options)

#         agents_now = list(obs.keys())
#         self.channel.reset(agents_now, seed=seed)
#         self.fault_model.reset(agents_now, seed=seed)

#         self._last_raw_obs = {a: obs[a] for a in obs}
#         obs = self._transform_obs(obs)
#         return obs, infos

#     def step(self, actions):
#         env_actions = self._transform_actions(actions)
#         obs, rewards, terms, truncs, infos = self.env.step(env_actions)
#         self.t += 1

#         self._last_raw_obs = {a: obs[a] for a in obs}
#         obs = self._transform_obs(obs)
#         return obs, rewards, terms, truncs, infos
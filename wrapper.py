from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np

from core import NetworkChannel, FaultModel, NetworkFaultConfig, DELIVERED_STATUSES, RECV_STATUSES


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

    Instrumentation
    ---------------
    For each delivered message the wrapper records a *sender-anchored* distortion
    (delivered payload vs. the payload the source actually sent, taken from the
    channel metadata), the real message age (t - creation time), the fraction of
    zeroed communication entries, and per-status delivery counts. These quantities
    are summed over an episode and finalized into per-episode metrics by
    :meth:`metrics`.
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
            # communication quality on receive side (sender-anchored)
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
            "delivered_bits": 0.0,

            # step-level bookkeeping
            "steps": 0,
        }
        # per-status delivery counters
        self.logs.update({f"status_{s}": 0 for s in RECV_STATUSES})

    def metrics(self) -> Dict[str, float]:
        """Finalize episode-level communication metrics from the running sums."""
        L = self.logs
        return {
            "comm_mse": L["mse_sum"] / max(1, L["mse_n"]),
            "comm_zero_frac": L["zero_sum"] / max(1, L["zero_n"]),
            "comm_age_mean": L["age_sum"] / max(1, L["age_n"]),
            "sent_msgs": float(L["sent_msgs"]),
            "delivered_real_msgs": float(L["delivered_real_msgs"]),
            "delivery_rate": L["delivered_real_msgs"] / max(1, L["sent_msgs"]),
            "loss_rate": 1.0 - L["delivered_real_msgs"] / max(1, L["sent_msgs"]),
            "delivered_bits": float(L["delivered_bits"]),
            "bits_per_delivered": L["delivered_bits"] / max(1, L["delivered_real_msgs"]),
            "steps": float(L["steps"]),
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
        return np.asarray(x, dtype=float).reshape(-1)

    def _safe_message_dim(self, agent: str, obs: Any = None) -> int:
        try:
            d = int(self.adapter.message_dim(self.env, agent, obs))
            return max(0, d)
        except TypeError:
            # adapters whose message_dim signature is (env, agent)
            try:
                d = int(self.adapter.message_dim(self.env, agent))
                return max(0, d)
            except Exception:
                return 0
        except Exception:
            return 0

    def _bits_per_entry(self) -> float:
        b = self.cfg.bandwidth_bits
        return float(b) if (b is not None and b > 0) else 32.0

    def _record_comm(self, meta: Dict[str, Any], delivered: np.ndarray) -> None:
        """Update running communication statistics from one recv() result."""
        status = meta.get("status", "none")
        self.logs[f"status_{status}"] = self.logs.get(f"status_{status}", 0) + 1

        rec = self._to_1d_float_array(delivered)

        # zero fraction is over every receive attempt (drops contribute all-zeros).
        if rec.size > 0:
            self.logs["zero_sum"] += int(np.sum(rec == 0.0))
            self.logs["zero_n"] += int(rec.size)

        if status in DELIVERED_STATUSES:
            self.logs["delivered_real_msgs"] += 1
            self.logs["age_sum"] += float(meta.get("age", 0))
            self.logs["age_n"] += 1
            self.logs["delivered_bits"] += rec.size * self._bits_per_entry()

            # sender-anchored distortion: delivered vs. what the source sent.
            src = meta.get("src_payload", None)
            if src is not None:
                src = self._to_1d_float_array(src)
                m = min(src.shape[0], rec.shape[0])
                if m > 0:
                    self.logs["mse_sum"] += float(np.mean((src[:m] - rec[:m]) ** 2))
                    self.logs["mse_n"] += 1

    # ---------------------------------------------------------
    # observation transform (receive side)
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
            msg_in, meta = self.channel.recv(
                receiver=agent,
                t=self.t,
                dim=msg_dim,
                agents=active_agents,
            )
            msg_in = self._to_1d_float_array(msg_in)

            self._record_comm(meta, msg_in)

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

    def _extract_raw_outgoing_message(
        self,
        agent: str,
        raw_obs: Any,
        env_action: Any,
        explicit_msg_out: Optional[np.ndarray],
    ) -> np.ndarray:
        msg = self.adapter.extract_outgoing_message(
            self.env, agent, raw_obs, env_action, explicit_msg_out
        )
        return self._to_1d_float_array(msg)

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

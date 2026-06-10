from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class SimpleDiscreteSpace:
    def __init__(self, n: int):
        self.n = int(n)
        self._rng = np.random.default_rng(0)

    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return int(self._rng.integers(0, self.n))


class WNTRParallelEnv:
    """
    Parallel-style normalized environment shim for WNTR.

    Each controllable link (pump or valve) becomes an agent:
        ctrl_<link_name>

    If the network has no pumps/valves, a single fallback agent is used:
        infra_operator

    Per step:
      - binary actions are applied to controllable links
      - one hydraulic control interval is simulated
      - a global feature vector is computed
      - the same global vector is returned to each active agent
      - a shared scalar reward is returned to each active agent
    """

    def __init__(
        self,
        inp_file: str,
        control_interval_seconds: int = 3600,
        horizon_steps: int = 24,
        pressure_threshold: float = 20.0,
        reward_control_penalty: float = 0.01,
    ):
        import wntr

        self.wntr = wntr
        self.inp_file = inp_file
        self.control_interval_seconds = int(control_interval_seconds)
        self.horizon_steps = int(horizon_steps)
        self.pressure_threshold = float(pressure_threshold)
        self.reward_control_penalty = float(reward_control_penalty)

        self._base_wn = None
        self.wn = None
        self._t = 0

        self._controllable_links = []
        self._agent_ids = []
        self._active_agents = []

        self._load_network()

    # ---------------------------------------------------------
    # env metadata
    # ---------------------------------------------------------

    def _load_network(self):
        self._base_wn = self.wntr.network.WaterNetworkModel(self.inp_file)
        self.wn = copy.deepcopy(self._base_wn)

        pump_names = list(self.wn.pump_name_list)
        valve_names = list(self.wn.valve_name_list)
        self._controllable_links = pump_names + valve_names

        if len(self._controllable_links) == 0:
            self._agent_ids = ["infra_operator"]
        else:
            self._agent_ids = [f"ctrl_{name}" for name in self._controllable_links]

        self._active_agents = list(self._agent_ids)

    @property
    def agents(self):
        return self._active_agents

    @property
    def possible_agents(self):
        return self._agent_ids

    def action_space(self, agent):
        # 0 = close/off, 1 = open/on
        return SimpleDiscreteSpace(2)

    def observation_space(self, agent):
        # Left as None because obs dimension is derived from the network simulation.
        return None

    # ---------------------------------------------------------
    # core internals
    # ---------------------------------------------------------

    def _reset_network(self, seed: Optional[int] = None):
        seed_everything(0 if seed is None else int(seed))
        self.wn = copy.deepcopy(self._base_wn)
        self._t = 0
        self._active_agents = list(self._agent_ids)

    def _apply_actions(self, actions: Dict[str, Any]) -> float:
        """
        Apply binary commands:
          cmd = 1 -> open/on
          cmd = 0 -> closed/off

        Returns:
          control_effort
        """
        if len(self._controllable_links) == 0:
            return 0.0

        effort = 0.0

        for link_name in self._controllable_links:
            agent_id = f"ctrl_{link_name}"
            cmd = int(actions.get(agent_id, 1))
            cmd = 1 if cmd != 0 else 0
            effort += abs(cmd - 1)

            link = self.wn.get_link(link_name)

            try:
                if cmd == 1:
                    link.initial_status = self.wntr.network.base.LinkStatus.Open
                else:
                    link.initial_status = self.wntr.network.base.LinkStatus.Closed
            except Exception:
                pass

        return float(effort)

    def _run_one_interval(self):
        """
        Simulate one control interval.

        EPANET writes intermediate temp files keyed by ``file_prefix``; when many
        environments run in parallel processes they must not share that prefix or
        the EPANET C library can crash on concurrent temp-file access. We give each
        run a process- and instance-unique prefix inside the system temp dir.
        """
        self.wn.options.time.duration = self.control_interval_seconds
        self.wn.options.time.hydraulic_timestep = self.control_interval_seconds
        self.wn.options.time.report_timestep = self.control_interval_seconds

        import os
        import tempfile
        prefix = os.path.join(tempfile.gettempdir(), f"wntr_{os.getpid()}_{id(self)}_{self._t}")

        try:
            sim = self.wntr.sim.EpanetSimulator(self.wn)
            results = sim.run_sim(file_prefix=prefix)
        except Exception:
            sim = self.wntr.sim.WNTRSimulator(self.wn)
            results = sim.run_sim()

        return results

    def _extract_global_features(self, results) -> np.ndarray:
        """
        Global numeric state vector built from:
          - node pressure
          - node head
          - link flowrate
        """
        feats = []

        try:
            pressure = results.node["pressure"].iloc[-1]
            feats.extend(np.asarray(pressure.values, dtype=np.float32).tolist())
        except Exception:
            pass

        try:
            head = results.node["head"].iloc[-1]
            feats.extend(np.asarray(head.values, dtype=np.float32).tolist())
        except Exception:
            pass

        try:
            flowrate = results.link["flowrate"].iloc[-1]
            feats.extend(np.asarray(flowrate.values, dtype=np.float32).tolist())
        except Exception:
            pass

        if len(feats) == 0:
            feats = [0.0]

        return np.asarray(feats, dtype=np.float32)

    def _compute_reward(self, results, control_effort: float) -> float:
        """
        Default reward:
          - penalize low pressure deficits
          - penalize control effort
        """
        low_pressure_penalty = 0.0

        try:
            pressure = results.node["pressure"].iloc[-1]
            deficit = np.maximum(
                self.pressure_threshold - np.asarray(pressure.values, dtype=float),
                0.0,
            )
            low_pressure_penalty = float(np.sum(deficit))
        except Exception:
            low_pressure_penalty = 0.0

        reward = -low_pressure_penalty - self.reward_control_penalty * float(control_effort)
        return float(reward)

    def _build_obs_dict(self, obs_vec: np.ndarray) -> Dict[str, np.ndarray]:
        if len(self._controllable_links) == 0:
            return {"infra_operator": obs_vec.copy()}
        return {a: obs_vec.copy() for a in self._agent_ids}

    def _build_info_dict(self) -> Dict[str, dict]:
        if len(self._controllable_links) == 0:
            return {"infra_operator": {}}
        return {a: {} for a in self._agent_ids}

    # ---------------------------------------------------------
    # public API
    # ---------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._reset_network(seed=seed)

        # Start with nominal control actions.
        nominal_actions = {a: 1 for a in self._agent_ids}
        self._apply_actions(nominal_actions)

        results = self._run_one_interval()
        obs_vec = self._extract_global_features(results)

        obs_dict = self._build_obs_dict(obs_vec)
        infos = self._build_info_dict()
        return obs_dict, infos

    def step(self, actions: Dict[str, Any]):
        control_effort = self._apply_actions(actions)
        results = self._run_one_interval()

        obs_vec = self._extract_global_features(results)
        reward = self._compute_reward(results, control_effort)

        self._t += 1
        done = self._t >= self.horizon_steps

        obs_dict = self._build_obs_dict(obs_vec)

        if len(self._controllable_links) == 0:
            rewards = {"infra_operator": reward}
            terms = {"infra_operator": bool(done)}
            truncs = {"infra_operator": False}
            infos = {"infra_operator": {}}
        else:
            rewards = {a: reward for a in self._agent_ids}
            terms = {a: bool(done) for a in self._agent_ids}
            truncs = {a: False for a in self._agent_ids}
            infos = {a: {} for a in self._agent_ids}

        self._active_agents = [] if done else list(self._agent_ids)
        return obs_dict, rewards, terms, truncs, infos

    def close(self):
        # WNTR does not usually need explicit closing.
        return None

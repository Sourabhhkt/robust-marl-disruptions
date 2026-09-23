"""
Cooperative multi-agent environments (Dec-POMDPs) for the Milestone 10 deep-MARL
baselines. They run over the same ``core.NetworkChannel`` disruption engine as the
rest of the suite, so a trained policy can be evaluated under message drop,
latency, bandwidth limits, spoofing, jamming, crashes, and Byzantine agents.

Common interface (a light, PettingZoo-flavoured parallel API; homogeneous agents
with parameter sharing):

    env.n_agents : int
    env.obs_dim  : int            per-agent observation size
    env.n_actions: int            discrete action count (shared across agents)
    env.state_dim: int            global state size (centralized critic / mixer)
    obs   = env.reset(seed)       -> (n_agents, obs_dim) float32
    obs, reward, done, info = env.step(actions)   # actions: (n_agents,) int
                                  reward is a single shared cooperative scalar
    s     = env.state()           -> (state_dim,) float32 global state
    mask  = env.alive_mask()      -> (n_agents,) bool (crashed/Byzantine handling)

ConsensusMARLEnv
----------------
Each agent holds a scalar state x_i and must reach team consensus. Every step it
observes summary statistics of the (impaired) messages it received and chooses,
from a small discrete set, *how to aggregate* them and *how far to move*: the
action selects a target (own / mean / median / trimmed-mean of received) and a
gain. The learning problem is to pick robust aggregation adaptively from the
observed communication conditions (e.g., switch to the median when the received
spread signals outliers), which no single fixed classical rule can do across
regimes. Byzantine agents (in the disruption regime) broadcast adversarial noise
and are excluded from the reward; the shared reward rewards honest-team consensus.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core import NetworkChannel, FaultModel, NetworkFaultConfig, DELIVERED_STATUSES
import synth_envs as SE


# Action factorization: target choice x gain  (shared by all agents).
_TARGETS = ["own", "mean", "median", "trimmed"]
_GAINS = [0.15, 0.35, 0.55]


def _trimmed(vals: np.ndarray, frac: float = 0.25) -> float:
    v = np.sort(np.asarray(vals, dtype=float))
    k = int(np.floor(frac * v.size))
    v = v[k: v.size - k] if (v.size - 2 * k) > 0 else v
    return float(np.mean(v))


def sample_disruption(rng: np.random.Generator) -> NetworkFaultConfig:
    """Domain-randomized impairment for the 'disrupt' training regime: pick one
    family and a random severity, matching the evaluation families."""
    fam = rng.choice(["clean", "msg_drop", "latency", "bandwidth", "spoof", "byzantine"])
    kw: Dict[str, Any] = {}
    byz = 0.0
    if fam == "msg_drop":
        kw["msg_drop_prob"] = float(rng.uniform(0.1, 0.7))
    elif fam == "latency":
        kw["base_latency_steps"] = int(rng.integers(1, 6))
    elif fam == "bandwidth":
        kw["bandwidth_bits"] = int(rng.integers(1, 5)); kw["quant_clip"] = (-3.0, 3.0)
    elif fam == "spoof":
        kw["spoof_prob"] = float(rng.uniform(0.1, 0.6)); kw["spoof_scale"] = 1.0
    elif fam == "byzantine":
        kw["byzantine_comm_corrupt_prob"] = 1.0; kw["spoof_scale"] = 2.0
        byz = float(rng.choice([0.1, 0.2, 0.3]))
    return NetworkFaultConfig(**kw), byz


class ConsensusMARLEnv:
    n_targets = len(_TARGETS)
    n_gains = len(_GAINS)

    def __init__(self, n: int = 12, steps: int = 40, topology: str = "ring_plus",
                 p_extra: float = 0.2, d: int = 1, regime: str = "clean",
                 reward: str = "shaped", action_type: str = "continuous",
                 gmax: float = 0.6):
        self.n_agents = int(n)
        self.steps = int(steps)
        self.topology = topology
        self.p_extra = p_extra
        self.d = int(d)
        self.regime = regime              # "clean" or "disrupt"
        self.reward_mode = reward         # "shaped" (per-step) or "terminal"
        # Continuous control (default; for IPPO/MAPPO/MADDPG): action = (blend, gain)
        # squashed to [0,1]; target interpolates median<->mean and the agent moves
        # 'gain' of the way toward it. Discrete control (action_type='discrete', for
        # the QMIX ablation only) bins the same (target, gain) choices.
        self.continuous = (action_type == "continuous")
        self.gmax = gmax
        self.act_dim = 2                  # (blend, gain)
        self.n_actions = self.n_targets * self.n_gains
        self.obs_dim = 9
        self.state_dim = self.n_agents * self.d + 2
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def _decode(self, a: int) -> Tuple[str, float]:
        return _TARGETS[a // self.n_gains], _GAINS[a % self.n_gains]

    def _disagreement(self, X: Optional[np.ndarray] = None) -> float:
        X = self.X if X is None else X
        h = X[self.honest]
        if h.shape[0] == 0:
            return 0.0
        return float(np.sum((h - h.mean(axis=0)) ** 2))

    def reset(self, seed: int = 0, cfg: Optional[NetworkFaultConfig] = None,
              byz_frac: Optional[float] = None) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        env_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(3)[2])
        n = self.n_agents
        self.agents = [str(i) for i in range(n)]
        self.adj = SE.build_graph(n, self.topology, env_rng, p_extra=self.p_extra)
        self.nbrs = [list(np.where(self.adj[i] > 0)[0]) for i in range(n)]

        # disruption: explicit override (evaluation), else per-episode by regime
        if cfg is not None:
            self.cfg, byz_frac = cfg, (byz_frac or 0.0)
        elif self.regime == "disrupt":
            self.cfg, byz_frac = sample_disruption(self._rng)
        else:
            self.cfg, byz_frac = NetworkFaultConfig(seed=seed), 0.0
        self.cfg.seed = int(seed)
        nb = int(round(byz_frac * n))
        self.byz = set(range(nb))
        self.honest = np.array([i not in self.byz for i in range(n)])

        self.X = env_rng.normal(0.0, 1.0, size=(n, self.d))
        self.byz_scale = 3.0
        self.channel = NetworkChannel(self.cfg); self.channel.reset(self.agents, seed=seed)
        self.fm = FaultModel(self.cfg); self.fm.reset(self.agents, seed=seed)
        self.t = 0
        self._V_prev = self._disagreement()
        self._V0 = max(self._V_prev, 1e-8)
        self._broadcast()
        return self._observe()

    def _broadcast(self):
        """Every agent sends its current state (Byzantine agents send noise)."""
        self.fm.begin_step(self.agents, self.t)
        for i in range(self.n_agents):
            if self.fm.is_crashed(self.agents[i], self.t):
                continue
            if i in self.byz:
                payload = self._rng.normal(0.0, self.byz_scale, size=self.d)
            else:
                payload = self.X[i].copy()
            for j in self.nbrs[i]:
                self.channel.send(self.agents[i], self.agents[j], payload, self.t,
                                  agents=self.agents, allow_byzantine_corrupt=True)

    def _received(self) -> List[List[float]]:
        out: List[List[float]] = []
        for j in range(self.n_agents):
            vals: List[float] = []
            if not self.fm.is_crashed(self.agents[j], self.t):
                got = self.channel.recv_all(self.agents[j], self.t, dim=self.d, agents=self.agents)
                for src, (payload, meta) in got.items():
                    if meta["status"] in DELIVERED_STATUSES:
                        v = np.asarray(self.fm.transform_observation(self.agents[j], payload, self.t),
                                       dtype=float).reshape(-1)[:self.d]
                        vals.append(float(v[0]))
            out.append(vals)
        self._last_recv = out
        return out

    def _observe(self) -> np.ndarray:
        recv = self._received()
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for j in range(self.n_agents):
            own = float(self.X[j, 0])
            vals = recv[j]
            if vals:
                a = np.asarray(vals)
                feat = [own, float(a.mean()), float(np.median(a)), _trimmed(a),
                        float(a.min()), float(a.max()), float(a.std()),
                        len(vals) / max(1, len(self.nbrs[j])), own - float(a.mean())]
            else:
                feat = [own, own, own, own, own, own, 0.0, 0.0, 0.0]
            obs[j] = np.asarray(feat, dtype=np.float32)
        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        recv = self._last_recv
        newX = self.X.copy()
        actions = np.asarray(actions)
        for i in range(self.n_agents):
            if i in self.byz or self.fm.is_crashed(self.agents[i], self.t):
                continue
            vals = recv[i]
            if not vals:
                continue
            arr = np.asarray(vals)
            if self.continuous:
                blend = 1.0 / (1.0 + np.exp(-float(actions[i, 0])))      # sigmoid
                gain = self.gmax / (1.0 + np.exp(-float(actions[i, 1])))
                target = blend * float(np.median(arr)) + (1 - blend) * float(arr.mean())
            else:
                tgt_name, gain = self._decode(int(actions[i]))
                if tgt_name == "own":
                    continue
                target = {"mean": arr.mean(), "median": np.median(arr),
                          "trimmed": _trimmed(arr)}[tgt_name]
            newX[i, 0] = self.X[i, 0] + gain * (float(target) - self.X[i, 0])
        self.X = newX
        self.t += 1

        V = self._disagreement()
        if self.reward_mode == "terminal":
            reward = 0.0
        else:  # shaped: normalized disagreement reduction
            reward = float((self._V_prev - V) / self._V0)
        self._V_prev = V

        done = self.t >= self.steps
        if done and self.reward_mode == "terminal":
            reward = float(-V / self._V0)
        if not done:
            self._broadcast()
            obs = self._observe()
        else:
            obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        info = {"disagreement": V, "V0": self._V0}
        return obs, reward, done, info

    def state(self) -> np.ndarray:
        s = np.zeros(self.state_dim, dtype=np.float32)
        s[:self.n_agents * self.d] = self.X.reshape(-1)
        s[-2] = self._disagreement()
        s[-1] = float(self.t) / self.steps
        return s

    def alive_mask(self) -> np.ndarray:
        return self.honest.copy()


class DispatchMARLEnv:
    """Economic dispatch by consensus as a cooperative Dec-POMDP. Each agent is a
    generator that adjusts its incremental cost lambda_i toward team agreement
    (which equalizes marginal costs = the optimum) while a balance feedback drives
    supply to demand. As in the consensus env, the action chooses how to aggregate
    the received (impaired) price messages and how far to move; the reward is the
    normalized reduction in the feasibility-aware dispatch regret."""
    n_targets = len(_TARGETS)
    n_gains = len(_GAINS)

    def __init__(self, n: int = 8, steps: int = 50, topology: str = "ring_plus",
                 p_extra: float = 0.3, regime: str = "clean", eps_balance: float = 0.12,
                 demand_frac: float = 0.6, hetero: float = 1.0, reward: str = "shaped",
                 action_type: str = "continuous"):
        self.n_agents = int(n); self.steps = int(steps)
        self.topology, self.p_extra = topology, p_extra
        self.regime, self.reward_mode = regime, reward
        self.eps_balance, self.demand_frac, self.hetero = eps_balance, demand_frac, hetero
        self.continuous = (action_type == "continuous")
        self.act_dim = 2
        self.n_actions = self.n_targets * self.n_gains
        self.obs_dim = 10
        self.state_dim = 2 * self.n_agents + 2
        self._rng = np.random.default_rng(0)

    def _decode(self, a):
        return _TARGETS[a // self.n_gains], _GAINS[a % self.n_gains]

    def _regret(self):
        import infra_envs as IE
        P = np.clip((self.lam - self.b) / (2 * self.a), 0.0, self.pmax)
        cost = float(np.sum(self.a * P * P + self.b * P))
        lag = cost + self.opt["lambda"] * (self.D - float(P.sum()))
        return max(0.0, (lag - self.opt["cost"]) / max(1e-6, abs(self.opt["cost"]))), P

    def reset(self, seed=0, cfg=None, byz_frac=None):
        import infra_envs as IE
        self._rng = np.random.default_rng(seed)
        env_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(5)[4])
        n = self.n_agents
        prob = IE.make_dispatch_problem(n, seed, demand_frac=self.demand_frac, hetero=self.hetero)
        self.a, self.b, self.pmax, self.D = prob["a"], prob["b"], prob["pmax"], prob["D"]
        self.opt = IE.central_optimum(prob)
        self.agents = [str(i) for i in range(n)]
        self.adj = SE.build_graph(n, self.topology, env_rng, p_extra=self.p_extra)
        self.W = IE.metropolis_weights(self.adj)
        self.nbrs = [list(np.where(self.adj[i] > 0)[0]) for i in range(n)]

        if cfg is not None:
            self.cfg, byz_frac = cfg, (byz_frac or 0.0)
        elif self.regime == "disrupt":
            self.cfg, byz_frac = sample_disruption(self._rng)
        else:
            self.cfg, byz_frac = NetworkFaultConfig(seed=seed), 0.0
        self.cfg.seed = int(seed)
        nb = int(round(byz_frac * n)); self.byz = set(range(nb))
        self.honest = np.array([i not in self.byz for i in range(n)])

        d_share = np.full(n, self.D / n)
        self.P = np.clip(d_share.copy(), 0.0, self.pmax)
        self.lam = 2.0 * self.a * self.P + self.b
        self.y = d_share - self.P
        self.byz_scale = 3.0
        self.channel = NetworkChannel(self.cfg); self.channel.reset(self.agents, seed=seed)
        self.fm = FaultModel(self.cfg); self.fm.reset(self.agents, seed=seed)
        self.t = 0
        self._reg_prev, _ = self._regret(); self._reg0 = max(self._reg_prev, 1e-6)
        self._broadcast(); return self._observe()

    def _broadcast(self):
        self.fm.begin_step(self.agents, self.t)
        for i in range(self.n_agents):
            if self.fm.is_crashed(self.agents[i], self.t):
                continue
            payload = (self._rng.normal(0.0, self.byz_scale, size=2) if i in self.byz
                       else np.array([self.lam[i], self.y[i]]))
            for j in self.nbrs[i]:
                self.channel.send(self.agents[i], self.agents[j], payload, self.t,
                                  agents=self.agents, allow_byzantine_corrupt=True)

    def _received(self):
        out = []
        for j in range(self.n_agents):
            lam_v, y_v, w_v = [], [], []
            if not self.fm.is_crashed(self.agents[j], self.t):
                got = self.channel.recv_all(self.agents[j], self.t, dim=2, agents=self.agents)
                for src, (payload, meta) in got.items():
                    if meta["status"] in DELIVERED_STATUSES:
                        v = np.asarray(self.fm.transform_observation(self.agents[j], payload, self.t),
                                       dtype=float).reshape(-1)[:2]
                        lam_v.append(float(v[0])); y_v.append(float(v[1])); w_v.append(float(self.W[j, int(src)]))
            out.append((lam_v, y_v, w_v))
        self._last_recv = out; return out

    def _observe(self):
        recv = self._received()
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for j in range(self.n_agents):
            lam_v, y_v, _ = recv[j]
            own_l, own_y = float(self.lam[j]), float(self.y[j])
            if lam_v:
                la = np.asarray(lam_v)
                feat = [own_l, float(la.mean()), float(np.median(la)), _trimmed(la), float(la.std()),
                        own_y, float(np.mean(y_v)), float(self.P[j] / max(1e-6, self.pmax[j])),
                        len(lam_v) / max(1, len(self.nbrs[j])), own_l - float(la.mean())]
            else:
                feat = [own_l, own_l, own_l, own_l, 0.0, own_y, own_y,
                        float(self.P[j] / max(1e-6, self.pmax[j])), 0.0, 0.0]
            obs[j] = np.asarray(feat, dtype=np.float32)
        return obs

    def step(self, actions):
        import infra_envs as IE
        recv = self._last_recv
        actions = np.asarray(actions)
        lam_new, y_new = self.lam.copy(), self.y.copy()
        for j in range(self.n_agents):
            if j in self.byz or self.fm.is_crashed(self.agents[j], self.t):
                continue
            lam_v, y_v, w_v = recv[j]
            if not lam_v:
                continue
            ws, wn = float(self.W[j, j]), w_v
            if self.continuous:
                blend = 1.0 / (1.0 + np.exp(-float(actions[j, 0])))      # median<->mean
                move = 1.2 / (1.0 + np.exp(-float(actions[j, 1])))       # move strength
                lam_mix = blend * IE.combine("median", self.lam[j], lam_v, ws, wn) \
                    + (1 - blend) * IE.combine("mean", self.lam[j], lam_v, ws, wn)
                y_mix = blend * IE.combine("median", self.y[j], y_v, ws, wn) \
                    + (1 - blend) * IE.combine("mean", self.y[j], y_v, ws, wn)
            else:
                tgt, g = self._decode(int(actions[j]))
                strat = "mean" if tgt in ("own", "mean") else tgt
                move = g / 0.35
                lam_mix = IE.combine(strat, self.lam[j], lam_v, ws, wn)
                y_mix = IE.combine(strat, self.y[j], y_v, ws, wn)
            lam_new[j] = self.lam[j] + move * (lam_mix - self.lam[j]) + self.eps_balance * self.y[j]
            y_new[j] = self.y[j] + move * (y_mix - self.y[j])
        self.lam = lam_new
        Pnew = np.clip((self.lam - self.b) / (2 * self.a), 0.0, self.pmax)
        self.y = y_new - (Pnew - self.P); self.P = Pnew
        self.t += 1

        reg, _ = self._regret()
        if self.reward_mode == "terminal":
            reward = 0.0
        else:
            reward = float((self._reg_prev - reg) / self._reg0)
        self._reg_prev = reg
        done = self.t >= self.steps
        if done and self.reward_mode == "terminal":
            reward = float(-reg / self._reg0)
        if not done:
            self._broadcast(); obs = self._observe()
        else:
            obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        return obs, reward, done, {"regret": reg, "reg0": self._reg0,
                                   "disagreement": float(np.var(self.lam[self.honest]) if self.honest.any() else 0.0)}

    def state(self):
        s = np.zeros(self.state_dim, dtype=np.float32)
        s[:self.n_agents] = self.lam; s[self.n_agents:2 * self.n_agents] = self.y
        s[-2] = self._reg_prev; s[-1] = float(self.t) / self.steps
        return s

    def alive_mask(self):
        return self.honest.copy()


def make_env(name: str, regime: str = "clean", **kw):
    """Environment factory."""
    if name in ("consensus", "consensus_marl"):
        return ConsensusMARLEnv(regime=regime, **kw)
    if name in ("dispatch", "dispatch_marl"):
        return DispatchMARLEnv(regime=regime, **kw)
    if name.startswith("mpe"):
        from .mpe_env import MPEMARLEnv
        return MPEMARLEnv(name=name, regime=regime, **kw)
    raise KeyError(f"unknown MARL env {name}")

"""
Compact cooperative-MARL algorithms with parameter sharing:

  - PPO (IPPO / MAPPO): on-policy actor-critic. IPPO uses a per-agent value
    function V(o_i); MAPPO uses a centralized critic V(s) over the global state.
    Shared categorical policy, clipped surrogate objective, GAE, entropy bonus.
  - QMIX: off-policy value decomposition. Shared per-agent Q-net + a monotonic
    mixing network over the global state, replay buffer, target networks,
    epsilon-greedy exploration, TD(0) loss.

Cooperative shared-reward setting; Byzantine / crashed agents are masked out of
the loss (they are not policy-controlled). The trainer (train.py) drives episodic
rollouts and calls store()/update().
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from .networks import Actor, GaussianActor, DetActor, CentralQ, Critic, QNet, QMixer


def _device(dev: str) -> torch.device:
    return torch.device("cuda" if (dev == "cuda" and torch.cuda.is_available()) else "cpu")


# ============================================================
# PPO (IPPO / MAPPO)
# ============================================================
class PPO:
    def __init__(self, obs_dim, n_actions, state_dim, n_agents, *, centralized=False,
                 continuous=False, act_dim=2, device="cpu", lr=3e-4, gamma=0.99,
                 lam=0.95, clip=0.2, epochs=4, minibatches=4, ent_coef=0.01,
                 vf_coef=0.5, seed=0):
        torch.manual_seed(seed)
        self.dev = _device(device)
        self.continuous = continuous
        self.act_dim = act_dim
        if continuous:
            self.actor = GaussianActor(obs_dim, act_dim).to(self.dev)
        else:
            self.actor = Actor(obs_dim, n_actions).to(self.dev)
        self.centralized = centralized
        self.critic = Critic(state_dim if centralized else obs_dim).to(self.dev)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) +
                                    list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.n_agents = n_agents
        self._ep: List[Dict[str, list]] = []
        self._cur: Optional[Dict[str, list]] = None

    @torch.no_grad()
    def act(self, obs, state, explore=True):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
        dist = self.actor.dist(o)
        if self.continuous:
            a = dist.sample() if explore else dist.mean
            logp = dist.log_prob(a).sum(-1)
            a_out = a.cpu().numpy()
        else:
            a = dist.sample() if explore else dist.probs.argmax(-1)
            logp = dist.log_prob(a)
            a_out = a.cpu().numpy()
        if self.centralized:
            s = torch.as_tensor(state, dtype=torch.float32, device=self.dev)
            v = self.critic(s).expand(self.n_agents)
        else:
            v = self.critic(o)
        return a_out, {"logp": logp.cpu().numpy(), "value": v.cpu().numpy()}

    def start_episode(self):
        self._cur = {k: [] for k in ("obs", "state", "act", "logp", "val", "rew", "alive")}

    def store(self, obs, state, actions, extra, reward, alive):
        c = self._cur
        c["obs"].append(obs); c["state"].append(state); c["act"].append(actions)
        c["logp"].append(extra["logp"]); c["val"].append(extra["value"])
        c["rew"].append(reward); c["alive"].append(alive)

    def end_episode(self, last_value=None):
        self._ep.append(self._cur); self._cur = None

    def _gae(self, rew, val, alive):
        T, n = val.shape
        adv = np.zeros((T, n), dtype=np.float32)
        last = np.zeros(n, dtype=np.float32)
        for t in reversed(range(T)):
            nextval = val[t + 1] if t + 1 < T else np.zeros(n, dtype=np.float32)
            nonterm = 1.0 if t + 1 < T else 0.0
            delta = rew[t] + self.gamma * nextval * nonterm - val[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + val
        return adv, ret

    def update(self):
        # flatten all (episode, step, agent) entries that are alive
        O, S, A, LP, ADV, RET = [], [], [], [], [], []
        for ep in self._ep:
            val = np.asarray(ep["val"], dtype=np.float32)             # (T,n)
            rew = np.asarray(ep["rew"], dtype=np.float32)             # (T,)
            alive = np.asarray(ep["alive"], dtype=bool)               # (T,n)
            rew2 = np.repeat(rew[:, None], val.shape[1], axis=1)
            adv, ret = self._gae(rew2, val, alive)
            obs = np.asarray(ep["obs"], dtype=np.float32)             # (T,n,obs)
            st = np.asarray(ep["state"], dtype=np.float32)            # (T,state)
            act = np.asarray(ep["act"])                              # (T,n)
            lp = np.asarray(ep["logp"], dtype=np.float32)            # (T,n)
            T, n = alive.shape
            for t in range(T):
                for k in range(n):
                    if not alive[t, k]:
                        continue
                    O.append(obs[t, k]); S.append(st[t]); A.append(act[t, k])
                    LP.append(lp[t, k]); ADV.append(adv[t, k]); RET.append(ret[t, k])
        self._ep = []
        if not O:
            return {"pi_loss": 0.0, "v_loss": 0.0}
        O = torch.as_tensor(np.asarray(O), dtype=torch.float32, device=self.dev)
        S = torch.as_tensor(np.asarray(S), dtype=torch.float32, device=self.dev)
        if self.continuous:
            A = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=self.dev)
        else:
            A = torch.as_tensor(np.asarray(A), dtype=torch.long, device=self.dev)
        LP = torch.as_tensor(np.asarray(LP), dtype=torch.float32, device=self.dev)
        ADV = torch.as_tensor(np.asarray(ADV), dtype=torch.float32, device=self.dev)
        RET = torch.as_tensor(np.asarray(RET), dtype=torch.float32, device=self.dev)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        N = O.shape[0]; mb = max(1, N // self.minibatches)
        pil, vll = 0.0, 0.0
        for _ in range(self.epochs):
            idx = torch.randperm(N, device=self.dev)
            for s in range(0, N, mb):
                b = idx[s:s + mb]
                dist = self.actor.dist(O[b])
                if self.continuous:
                    logp = dist.log_prob(A[b]).sum(-1)
                    ent = dist.entropy().sum(-1).mean()
                else:
                    logp = dist.log_prob(A[b])
                    ent = dist.entropy().mean()
                ratio = torch.exp(logp - LP[b])
                s1 = ratio * ADV[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[b]
                pi_loss = -torch.min(s1, s2).mean() - self.ent_coef * ent
                v = self.critic(S[b] if self.centralized else O[b])
                v_loss = ((v - RET[b]) ** 2).mean()
                loss = pi_loss + self.vf_coef * v_loss
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) +
                                         list(self.critic.parameters()), 0.5)
                self.opt.step()
                pil += float(pi_loss.item()); vll += float(v_loss.item())
        return {"pi_loss": pil, "v_loss": vll}

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])


# ============================================================
# QMIX
# ============================================================
class QMIX:
    def __init__(self, obs_dim, n_actions, state_dim, n_agents, *, device="cpu",
                 lr=5e-4, gamma=0.99, buffer=20000, batch=256, target_every=200,
                 eps_start=1.0, eps_end=0.05, eps_decay=8000, seed=0):
        torch.manual_seed(seed)
        self.dev = _device(device)
        self.q = QNet(obs_dim, n_actions).to(self.dev)
        self.qt = QNet(obs_dim, n_actions).to(self.dev)
        self.mix = QMixer(n_agents, state_dim).to(self.dev)
        self.mixt = QMixer(n_agents, state_dim).to(self.dev)
        self.qt.load_state_dict(self.q.state_dict()); self.mixt.load_state_dict(self.mix.state_dict())
        self.opt = torch.optim.Adam(list(self.q.parameters()) + list(self.mix.parameters()), lr=lr)
        self.gamma, self.batch = gamma, batch
        self.n_actions, self.n_agents = n_actions, n_agents
        self.target_every = target_every
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.buf: List[tuple] = []; self.buf_cap = buffer
        self.steps = 0; self._rng = np.random.default_rng(seed)

    def epsilon(self):
        f = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, state, explore=True):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
        q = self.q(o)                                    # (n, A)
        greedy = q.argmax(-1).cpu().numpy()
        if explore:
            eps = self.epsilon()
            rand = self._rng.random(self.n_agents) < eps
            ra = self._rng.integers(0, self.n_actions, self.n_agents)
            a = np.where(rand, ra, greedy)
        else:
            a = greedy
        return a.astype(np.int64), {}

    def store(self, obs, state, actions, extra, reward, alive, next_obs, next_state, done):
        self.buf.append((obs.astype(np.float32), state.astype(np.float32),
                         actions.astype(np.int64), float(reward),
                         next_obs.astype(np.float32), next_state.astype(np.float32),
                         float(done), alive.astype(np.float32)))
        if len(self.buf) > self.buf_cap:
            self.buf.pop(0)
        self.steps += 1

    def update(self):
        if len(self.buf) < self.batch:
            return {"q_loss": 0.0, "eps": self.epsilon()}
        idx = self._rng.integers(0, len(self.buf), self.batch)
        O, S, A, R, NO, NS, D, AL = zip(*[self.buf[i] for i in idx])
        O = torch.as_tensor(np.stack(O), device=self.dev)         # (B,n,obs)
        S = torch.as_tensor(np.stack(S), device=self.dev)         # (B,state)
        A = torch.as_tensor(np.stack(A), device=self.dev)         # (B,n)
        R = torch.as_tensor(np.asarray(R, dtype=np.float32), device=self.dev)
        NO = torch.as_tensor(np.stack(NO), device=self.dev)
        NS = torch.as_tensor(np.stack(NS), device=self.dev)
        D = torch.as_tensor(np.asarray(D, dtype=np.float32), device=self.dev)
        AL = torch.as_tensor(np.stack(AL), device=self.dev)       # (B,n)

        q = self.q(O).gather(-1, A.unsqueeze(-1)).squeeze(-1)      # (B,n)
        q = q * AL                                                # mask dead agents
        qtot = self.mix(q, S)                                     # (B,)
        with torch.no_grad():
            nq = self.qt(NO)                                       # (B,n,A)
            nq_max = nq.max(-1).values * AL
            ntot = self.mixt(nq_max, NS)
            target = R + self.gamma * (1 - D) * ntot
        loss = ((qtot - target) ** 2).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(self.q.parameters()) + list(self.mix.parameters()), 5.0)
        self.opt.step()
        if self.steps % self.target_every == 0:
            self.qt.load_state_dict(self.q.state_dict()); self.mixt.load_state_dict(self.mix.state_dict())
        return {"q_loss": float(loss.item()), "eps": self.epsilon()}

    def state_dict(self):
        return {"q": self.q.state_dict(), "mix": self.mix.state_dict()}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd["q"]); self.mix.load_state_dict(sd["mix"])
        self.qt.load_state_dict(self.q.state_dict()); self.mixt.load_state_dict(self.mix.state_dict())


# ============================================================
# MADDPG (continuous, centralized critic) -- for the continuous synthetic tasks
# ============================================================
class MADDPG:
    def __init__(self, obs_dim, act_dim, state_dim, n_agents, *, device="cpu",
                 lr=1e-3, gamma=0.99, tau=0.01, buffer=50000, batch=256,
                 noise=0.2, seed=0):
        torch.manual_seed(seed)
        self.dev = _device(device)
        self.continuous = True
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.actor = DetActor(obs_dim, act_dim).to(self.dev)
        self.actor_t = DetActor(obs_dim, act_dim).to(self.dev)
        self.critic = CentralQ(state_dim, n_agents * act_dim).to(self.dev)
        self.critic_t = CentralQ(state_dim, n_agents * act_dim).to(self.dev)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau, self.batch, self.noise = gamma, tau, batch, noise
        self.buf: List[tuple] = []; self.buf_cap = buffer
        self.steps = 0; self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, state, explore=True):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
        a = self.actor(o).cpu().numpy()
        if explore:
            a = a + self._rng.normal(0, self.noise, size=a.shape)
            a = np.clip(a, -1.0, 1.0)
        return a.astype(np.float32), {}

    def store(self, obs, state, actions, extra, reward, alive, next_obs, next_state, done):
        self.buf.append((obs.astype(np.float32), state.astype(np.float32),
                         np.asarray(actions, np.float32), float(reward),
                         next_obs.astype(np.float32), next_state.astype(np.float32),
                         float(done), alive.astype(np.float32)))
        if len(self.buf) > self.buf_cap:
            self.buf.pop(0)
        self.steps += 1

    def _soft(self, net, net_t):
        for p, pt in zip(net.parameters(), net_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def update(self):
        if len(self.buf) < self.batch:
            return {"q_loss": 0.0}
        idx = self._rng.integers(0, len(self.buf), self.batch)
        O, S, A, R, NO, NS, D, AL = zip(*[self.buf[i] for i in idx])
        B = self.batch
        O = torch.as_tensor(np.stack(O), device=self.dev)          # (B,n,obs)
        S = torch.as_tensor(np.stack(S), device=self.dev)
        A = torch.as_tensor(np.stack(A), device=self.dev)          # (B,n,act)
        R = torch.as_tensor(np.asarray(R, np.float32), device=self.dev)
        NO = torch.as_tensor(np.stack(NO), device=self.dev)
        NS = torch.as_tensor(np.stack(NS), device=self.dev)
        D = torch.as_tensor(np.asarray(D, np.float32), device=self.dev)
        AL = torch.as_tensor(np.stack(AL), device=self.dev)        # (B,n)

        with torch.no_grad():
            na = self.actor_t(NO)                                  # (B,n,act)
            qt = self.critic_t(NS, na.reshape(B, -1))
            y = R + self.gamma * (1 - D) * qt
        q = self.critic(S, A.reshape(B, -1))
        c_loss = ((q - y) ** 2).mean()
        self.opt_c.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.opt_c.step()

        # actor: predicted actions for honest agents, buffer actions for Byzantine
        pred = self.actor(O)                                       # (B,n,act)
        m = AL.unsqueeze(-1)
        joint = (m * pred + (1 - m) * A).reshape(B, -1)
        a_loss = -self.critic(S, joint).mean()
        self.opt_a.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt_a.step()

        self._soft(self.actor, self.actor_t); self._soft(self.critic, self.critic_t)
        return {"q_loss": float(c_loss.item())}

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"]); self.critic.load_state_dict(sd["critic"])
        self.actor_t.load_state_dict(self.actor.state_dict())


def make_algo(name, obs_dim, n_actions, state_dim, n_agents, device="cpu", seed=0,
              continuous=False, act_dim=2):
    """Algorithm factory. Continuous methods (ippo/mappo/maddpg) use act_dim; the
    discrete value method (qmix) uses n_actions and is intended for the
    discrete-action MPE tasks only (the continuous synthetic tasks use the
    continuous methods, per the M9 discrete-action caveat)."""
    name = name.lower()
    if name == "ippo":
        return PPO(obs_dim, n_actions, state_dim, n_agents, centralized=False,
                   continuous=continuous, act_dim=act_dim, device=device, seed=seed)
    if name == "mappo":
        return PPO(obs_dim, n_actions, state_dim, n_agents, centralized=True,
                   continuous=continuous, act_dim=act_dim, device=device, seed=seed)
    if name == "maddpg":
        return MADDPG(obs_dim, act_dim, state_dim, n_agents, device=device, seed=seed)
    if name == "qmix":
        return QMIX(obs_dim, n_actions, state_dim, n_agents, device=device, seed=seed)
    raise KeyError(f"unknown algo {name}")

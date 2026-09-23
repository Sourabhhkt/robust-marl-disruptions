"""Compact networks for the Milestone 10 MARL baselines (parameter-shared)."""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes, act=nn.Tanh, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Shared categorical policy over discrete actions."""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, n_actions])

    def forward(self, obs):
        return self.net(obs)                      # logits

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.forward(obs))


class GaussianActor(nn.Module):
    """Shared diagonal-Gaussian policy over a continuous action vector (IPPO/MAPPO
    on the continuous synthetic tasks). Raw actions are unbounded; the environment
    squashes them (sigmoid/tanh) into its control ranges."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.mu = mlp([obs_dim, hidden, hidden, act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs):
        return self.mu(obs)

    def dist(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return torch.distributions.Normal(mu, std)


class DetActor(nn.Module):
    """Deterministic squashed policy for MADDPG (tanh-bounded raw action)."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim])

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class CentralQ(nn.Module):
    """Centralized critic Q(state, joint_action) for MADDPG."""
    def __init__(self, state_dim, joint_act_dim, hidden=128):
        super().__init__()
        self.net = mlp([state_dim + joint_act_dim, hidden, hidden, 1])

    def forward(self, state, joint_act):
        return self.net(torch.cat([state, joint_act], dim=-1)).squeeze(-1)


class Critic(nn.Module):
    """Value head. For IPPO the input is the per-agent observation; for MAPPO it
    is the global state (centralized critic)."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, 1])

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QNet(nn.Module):
    """Shared per-agent action-value network for QMIX."""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, n_actions])

    def forward(self, obs):
        return self.net(obs)                      # (..., n_actions)


class QMixer(nn.Module):
    """Monotonic mixing network (hypernetwork-generated non-negative weights), so
    the joint Q is monotone in each agent's Q -- the QMIX consistency condition."""
    def __init__(self, n_agents, state_dim, embed=32):
        super().__init__()
        self.n = n_agents
        self.embed = embed
        self.hyp_w1 = nn.Linear(state_dim, n_agents * embed)
        self.hyp_w2 = nn.Linear(state_dim, embed)
        self.hyp_b1 = nn.Linear(state_dim, embed)
        self.hyp_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(),
                                    nn.Linear(embed, 1))

    def forward(self, agent_qs, state):
        # agent_qs: (B, n)   state: (B, state_dim)
        B = agent_qs.shape[0]
        w1 = torch.abs(self.hyp_w1(state)).view(B, self.n, self.embed)
        b1 = self.hyp_b1(state).view(B, 1, self.embed)
        h = torch.relu(torch.bmm(agent_qs.view(B, 1, self.n), w1) + b1)   # (B,1,embed)
        w2 = torch.abs(self.hyp_w2(state)).view(B, self.embed, 1)
        b2 = self.hyp_b2(state).view(B, 1, 1)
        y = torch.bmm(h, w2) + b2                                          # (B,1,1)
        return y.view(B)

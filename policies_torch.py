# policies_torch.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class SharedActorCritic(nn.Module):
  def __init__(self, obs_dim, n_actions, hidden=128):
    super().__init__()
    self.body = nn.Sequential(
                  nn.Linear(obs_dim, hidden), nn.ReLU(),
                  nn.Linear(hidden, hidden), nn.ReLU(),
                  )
    self.actor = nn.Linear(hidden, n_actions)
    self.critic = nn.Linear(hidden, 1)

  def act(self, obs):
    x = torch.as_tensor(obs, d_type=torch.float32)
    h = self.body(x)
    dist = Categorical(logits=self.actor(h))
    action = dist.sample()
    return action.item(), dist.log_prob(action), self.critic(h)

  def value(self, obs):
    x = torch.as_tensor(obs, dtype=torch.float32)
    return self.critic(self.body(x))

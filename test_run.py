import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=pd.errors.SettingWithCopyWarning,
)

from core import NetworkFaultConfig
from env_shims import Grid2OpParallelEnv
from wrapper import InstrumentedUniversalCommFaultWrapper
from adapter import Grid2OpCommAdapter


def make_grid2op_adapter(last_k: int = 8):
    def slice_fn(obs):
        arr = np.asarray(obs, dtype=float).reshape(-1)
        k = min(last_k, arr.shape[0])
        return arr[-k:]

    def replace_fn(obs, msg):
        arr = np.asarray(obs, dtype=float).copy().reshape(-1)
        k = min(msg.shape[0], arr.shape[0])
        arr[-k:] = msg[:k]
        return arr

    return Grid2OpCommAdapter(
        agents_list_fn=lambda env: env.possible_agents,
        slice_fn=slice_fn,
        replace_fn=replace_fn,
    )


env = Grid2OpParallelEnv(env_name="l2rpn_case14_sandbox", max_steps=5)
adapter = make_grid2op_adapter(last_k=8)
cfg = NetworkFaultConfig(seed=1, msg_drop_prob=0.1)

wrapped = InstrumentedUniversalCommFaultWrapper(env, adapter, cfg)
obs, infos = wrapped.reset(seed=123)

import torch
from policies_torch import SharedActorCritic

sample_agent = wrapped.agents[0]
obs_dim = len(np.asarray(obs[sample_agent]).reshape(-1))

ACTION_SET = [
    wrapped.action_space(sample_agent)({}),
    wrapped.action_space(sample_agent).sample(),
]

n_actions = len(ACTION_SET)

policy = SharedActorCritic(obs_dim, n_actions)
policy.eval()

while wrapped.agents:
    actions = {}

    for a in wrapped.agents:
        idx, logp, value = policy.act(obs[a])
        actions[a] = ACTION_SET[idx]

    obs, rewards, terms, truncs, infos = wrapped.step(actions)

    if (terms and all(terms.values())) or (truncs and all(truncs.values())):
        break

print(wrapped.logs)
wrapped.close()

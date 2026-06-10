# Guide to adding a new benchmark

Adding a benchmark to the suite means wiring it through the same three layers the
rest of the code uses (see `architecture.md`): an environment shim, a
communication adapter, and (optionally) a task entry in the experiment runner.
Two existing benchmarks are good references: `WNTRParallelEnv` in `env_shims.py`
(a wrapped real simulator) and the synthetic environments in `synth_envs.py`
(communication-native tasks).

---

## A. Environment layer: provide a normalized parallel API

The wrapper expects the benchmark environment to expose this interface:

- `reset(seed=None, options=None) -> (obs_dict, infos_dict)`
- `step(action_dict) -> (obs_dict, rewards_dict, terms_dict, truncs_dict, infos_dict)`
- `agents`
- `possible_agents`
- `action_space(agent)`
- `observation_space(agent)`
- `close()`

If the benchmark already has this interface (for example a PettingZoo
`parallel_env`), no shim is needed. Otherwise, create a small shim like
`WNTRParallelEnv` that:

- converts the benchmark's native reset/step format into dicts keyed by agent id,
- defines the agent names,
- maps the action spaces, and
- converts observations into a representation your adapter can process.

---

## B. Communication layer: define an adapter

This is the most important benchmark-specific decision. Answer:

1. Who are the agents?
2. What counts as the outgoing message?
3. Where does the incoming message get inserted?
4. What is the message dimension?
5. Does the benchmark use explicit messages or implicit observation-slice
   communication?

### Case 1: Observation-slice communication
If communication is part of the observation/state vector, use
`ObsSliceCommAdapter` and define `agents_list_fn(env)`, `slice_fn(obs)`, and
`replace_fn(obs, msg)`. This is how WNTR and the wrapped MPE benchmarks are
handled.

### Case 2: Explicit messages
If the action naturally includes a separate message component, configure an
adapter that receives `(env_action, msg_out)`, returns the explicit message via
`extract_outgoing_message(...)`, and injects the received message appropriately.

### Case 3: Generic dict-based multi-agent benchmark
If the environment is already dict-based and you only want to attach messages to
observations, `DictMAAdapter` is enough.

### Example (observation-slice, as used for WNTR)

```python
import numpy as np
from adapter import ObsSliceCommAdapter

def make_my_adapter(last_k=8):
    def slice_fn(obs):
        arr = np.asarray(obs, dtype=float).reshape(-1)
        return arr[-last_k:]

    def replace_fn(obs, msg):
        arr = np.asarray(obs, dtype=float).copy().reshape(-1)
        k = min(msg.shape[0], arr.shape[0])
        arr[-k:] = msg[:k]
        return arr

    return ObsSliceCommAdapter(
        agents_list_fn=lambda env: env.possible_agents,
        slice_fn=slice_fn,
        replace_fn=replace_fn,
    )
```

---

## C. Optional: register a task for the experiment runner

To include the benchmark in sweeps and figures, add a `run_*` function and an
entry to the `TASKS` registry in `baselines.py`. A task is
`run(cfg, seed, **env_overrides) -> record`, where `record` is a flat dict of
scalars (the runner aggregates these across seeds). The `_run_wrapped` helper in
`baselines.py` already drives a wrapped env with a policy and returns the
communication metrics; follow `run_wntr` as a template. Once registered, the
benchmark can be swept with `runner.run_sweep` and plotted with `analysis.py`.

---

## Minimal checklist

1. Create or reuse an environment shim with the dict-based parallel API above.
2. Create an adapter that defines the message interface.
3. (For sweeps) add a `run_*` task and a `TASKS` entry in `baselines.py`.
4. Otherwise, wrap the env directly and run a short rollout to inspect metrics.

---

## Standalone template (wrap and roll out directly)

```python
import numpy as np
from core import NetworkFaultConfig
from wrapper import InstrumentedUniversalCommFaultWrapper
from adapter import ObsSliceCommAdapter  # or DictMAAdapter / custom adapter
from env_shims import WNTRParallelEnv     # or your own shim

def make_my_adapter():
    def slice_fn(obs):
        arr = np.asarray(obs, dtype=float).reshape(-1)
        return arr[-4:]

    def replace_fn(obs, msg):
        arr = np.asarray(obs, dtype=float).copy().reshape(-1)
        arr[-4:] = msg[:4]
        return arr

    return ObsSliceCommAdapter(
        agents_list_fn=lambda env: env.possible_agents,
        slice_fn=slice_fn,
        replace_fn=replace_fn,
    )

env = WNTRParallelEnv("Net3.inp", horizon_steps=24)
adapter = make_my_adapter()
cfg = NetworkFaultConfig(seed=1, msg_drop_prob=0.1)

wrapped = InstrumentedUniversalCommFaultWrapper(env, adapter, cfg)
obs, infos = wrapped.reset(seed=123)

while wrapped.agents:
    actions = {a: wrapped.action_space(a).sample() for a in wrapped.agents}
    obs, rewards, terms, truncs, infos = wrapped.step(actions)
    if (terms and all(terms.values())) or (truncs and all(truncs.values())):
        break

print(wrapped.metrics())
wrapped.close()
```

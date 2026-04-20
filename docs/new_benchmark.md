# Guide to adapting `test_run.py` to a New Benchmark

## Purpose of `test_run(1).py`

`test_run(1).py` is a minimal smoke test for the communication-fault framework. It does four things:

1. creates a benchmark environment,
2. defines how communication is represented inside that benchmark,
3. wraps the environment with a generic communication-fault layer, and
4. runs a short random-action rollout while collecting instrumentation logs.

In the current version, the benchmark is **Grid2Op**.

---

## What needs to change for a new benchmark?

To support a new benchmark, need to think about **three layers**.

## A. Environment layer: provide a normalized parallel API

The wrapper expects the benchmark environment to expose something close to this interface:

- `reset(seed=None, options=None) -> (obs_dict, infos_dict)`
- `step(action_dict) -> (obs_dict, rewards_dict, terms_dict, truncs_dict, infos_dict)`
- `agents`
- `possible_agents`
- `action_space(agent)`
- `observation_space(agent)`
- `close()`

If your benchmark already has this interface, you may not need a shim.

If it does not, create a small shim similar to `Grid2OpParallelEnv` that:

- converts the benchmark’s native reset/step format into dicts keyed by agent id,
- defines agent names,
- maps action spaces, and
- converts observations into a representation your adapter can process.

### Typical edits

For a new benchmark, you would usually replace:

```python
from env_shims import Grid2OpParallelEnv
env = Grid2OpParallelEnv(...)
```

with something like:

```python
from env_shims import MyBenchmarkParallelEnv
env = MyBenchmarkParallelEnv(...)
```

---

## B. Communication layer: define an adapter

This is the most important benchmark-specific modification.

You need to decide:

1. **Who are the agents?**
2. **What counts as the outgoing message?**
3. **Where does the incoming message get inserted?**
4. **What is the message dimension?**
5. **Does the benchmark use explicit messages or implicit observation-state communication?**

### Case 1: Observation-slice communication

If the new benchmark represents communication as part of the observation/state vector, use `ObsSliceCommAdapter` (or a subclass like the current Grid2Op adapter).

You must define:

- `agents_list_fn(env)`
- `slice_fn(obs)`
- `replace_fn(obs, msg)`

This is enough when the message is a slice of the observation.

### Case 2: Explicit messages

If actions naturally include a separate message component, implement or configure an adapter that:

- gets `(env_action, msg_out)`,
- returns the explicit message through `extract_outgoing_message(...)`, and
- injects the received message appropriately.

### Case 3: Generic dict-based multi-agent benchmark

If your environment is already dict-based and you just want to append/attach messages to observations, `DictMAAdapter` may be enough.

### Typical edits

For a new benchmark, you would usually replace:

```python
adapter = make_grid2op_adapter(last_k=8)
```

with either:

- a new adapter factory, or
- a direct adapter instance for the new benchmark.

For example:

```python
def make_my_benchmark_adapter():
    def slice_fn(obs):
        arr = np.asarray(obs, dtype=float).reshape(-1)
        return arr[10:18]

    def replace_fn(obs, msg):
        arr = np.asarray(obs, dtype=float).copy().reshape(-1)
        arr[10:18] = msg
        return arr

    return ObsSliceCommAdapter(
        agents_list_fn=lambda env: env.possible_agents,
        slice_fn=slice_fn,
        replace_fn=replace_fn,
    )
```

---

## Minimal checklist for adding a new benchmark

1. **Create or reuse an environment shim** so the benchmark exposes dict-based multi-agent reset/step behavior.
2. **Create an adapter** that defines the message interface.
3. **Instantiate `NetworkFaultConfig`** with the impairments you want.
4. **Wrap the environment** with `InstrumentedUniversalCommFaultWrapper`.
5. **Run a short rollout** and inspect `wrapped.logs`.

---

## Practical template for a new benchmark

```python
from core import NetworkFaultConfig
from wrapper import InstrumentedUniversalCommFaultWrapper
from adapter import ObsSliceCommAdapter  # or DictMAAdapter / custom adapter
from env_shims import MyBenchmarkParallelEnv


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


env = MyBenchmarkParallelEnv(...)
adapter = make_my_adapter()
cfg = NetworkFaultConfig(seed=1, msg_drop_prob=0.1)

wrapped = InstrumentedUniversalCommFaultWrapper(env, adapter, cfg)
obs, infos = wrapped.reset(seed=123)

while wrapped.agents:
    actions = {a: wrapped.action_space(a).sample() for a in wrapped.agents}
    obs, rewards, terms, truncs, infos = wrapped.step(actions)
    if (terms and all(terms.values())) or (truncs and all(truncs.values())):
        break

print(wrapped.logs)
wrapped.close()
```


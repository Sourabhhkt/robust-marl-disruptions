# Architecture

This repository is organized around a simple idea:

> Keep **generic disruption logic** separate from **benchmark-specific translation logic**.

This makes it possible to apply the same communication and agent-fault models across different RL and MARL benchmarks without rewriting the fault machinery each time.

---

## High-Level Design

The codebase is split into four layers:

1. **`core.py`**  
   Benchmark-agnostic disruption engine

2. **`adapter.py`**  
   Benchmark-specific translation layer

3. **`env_shims.py`**  
   Benchmark normalization layer

4. **`wrapper.py`**  
   Universal orchestration layer

Together, these layers let a single wrapper apply the same network and fault model to environments such as Grid2Op and WNTR, even though those environments represent actions, observations, and communication differently.

---

## Design Principle

The central design principle is:

- **`core.py`** decides **how faults happen**
- **`adapter.py`** decides **what those faults act on for a given benchmark**
- **`env_shims.py`** makes every benchmark look like the same type of environment
- **`wrapper.py`** decides **when to call each part during reset and step**

This separation keeps the system modular and easier to extend.

---

## Layer 1: `core.py`

`core.py` contains the benchmark-independent disruption engine.

It should not need to know whether the benchmark is:
- Grid2Op
- WNTR
- PettingZoo
- SMAC
- Overcooked
- MAPF
- or any other environment

### Main responsibilities

#### 1. `NetworkFaultConfig`
A shared configuration object containing disruption settings such as:
- message drop probability
- latency and jitter
- duplication and reordering
- quantization and bandwidth caps
- replay and spoofing
- jamming and partitioning
- crash probability
- sensor noise
- actuator faults
- byzantine corruption

This object is the main user-facing way to configure a fault scenario.

#### 2. `NetworkChannel`
A simulated communication channel that handles message delivery and communication faults.

Typical responsibilities:
- queue outgoing messages
- schedule delivery under latency/jitter
- drop or duplicate messages
- reorder messages
- cap message dimension
- quantize payload values
- jam or partition receivers
- replay old messages
- spoof incoming payloads

`NetworkChannel` models what happens **between agents**.

#### 3. `FaultModel`
A generic agent-local fault engine.

Typical responsibilities:
- crash agents temporarily
- zero or distort observations during crash
- add observation noise
- modify actions under actuator faults
- replace actions with noop or random actions
- inject byzantine action corruption

`FaultModel` models what happens **at the agent itself**.

---

## Layer 2: `adapter.py`

`adapter.py` contains the benchmark-specific translation logic.

This layer is necessary because different benchmarks represent actions, observations, and communication in very different ways.

### Why adapters are needed

Different benchmarks can have very different semantics:

- In one benchmark, the outgoing message may be explicitly attached to the action.
- In another benchmark, the message may be represented as a slice of the observation.
- In one benchmark, a safe noop action may be integer `0`.
- In another benchmark, a safe noop may require a benchmark-specific object such as `action_space({})`.

A universal wrapper cannot infer these semantics on its own. The adapter exists to explain them.

### Main responsibilities

A benchmark adapter should answer the following questions:

#### 1. How is the action interpreted?
For example:
- is the action already the real environment action?
- is the action a tuple `(env_action, msg_out)`?
- is part of the action communication and part control?

#### 2. What is the message dimension?
For a given agent and observation, how many numbers make up the communication payload?

#### 3. How do we extract the outgoing message?
Examples:
- use the explicitly provided message from the action
- extract the last `k` values of the observation
- extract some benchmark-specific communication state

#### 4. How do we inject an incoming message?
Examples:
- concatenate the received message to the observation
- place the received message in a dict field
- overwrite a slice of the observation vector

#### 5. What is a safe noop action?
This is benchmark-dependent and must not be assumed to be the same for all environments.

#### 6. How do we sample a random action?
This may also be benchmark-specific.

---

## Layer 3: `env_shims.py`

`env_shims.py` provides normalized environment wrappers.

The goal is to make every supported benchmark expose the same external API, regardless of how the original benchmark is written.

### Normalized environment API

Every env shim should expose something like:

```python
obs_dict, infos = env.reset(seed=seed, options=options)
obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

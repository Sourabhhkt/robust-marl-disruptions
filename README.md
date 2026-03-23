# Universal Communication-Fault Wrapper for RL Benchmarks

This repository provides a **benchmark-agnostic framework** for injecting **communication disruptions** and **agent-level faults** into reinforcement learning (RL) and multi-agent reinforcement learning (MARL) benchmarks.

The main goal is to separate:

- **generic disruption logic** from
- **benchmark-specific translation logic**

so that the same universal wrapper can support heterogeneous environments such as:

- **Grid2Op**
- **WNTR**
- and, with additional adapters, benchmarks such as PettingZoo, SMAC, Overcooked, MAPF, Flatland, and others.


## Motivation

Distributed and multi-agent systems often operate under unreliable conditions:

- messages can be delayed or dropped
- communication may be quantized or bandwidth-limited
- agents may crash temporarily
- observations may become noisy
- actuators may fail
- malicious or byzantine behavior may occur

This repository addresses that problem by introducing a layered design:

1. **`core.py`** implements generic fault and network logic
2. **`adapter.py`** explains how a particular benchmark represents messages, actions, and observations
3. **`env_shims.py`** normalizes benchmark APIs into a common dict-based interface
4. **`wrapper.py`** orchestrates the interaction between the environment, the adapter, and the fault logic

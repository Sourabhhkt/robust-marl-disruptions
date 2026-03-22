# Faults and Disruptions

This document describes the disruption and fault mechanisms supported by the framework.

The goal of the framework is to inject controlled failures into RL and MARL benchmarks in a benchmark-agnostic way. These disruptions are configured through `NetworkFaultConfig` and are implemented primarily in `core.py` through:

- `NetworkChannel`
- `FaultModel`

Broadly, faults fall into two categories:

1. **Communication faults**  
   These affect messages moving between agents.

2. **Agent-local faults**  
   These affect what an agent senses or how it acts.

---

# 1. Configuration Overview

All disruptions are controlled through a shared configuration object.

Example:

```python
cfg = NetworkFaultConfig(
    seed=1,

    # communication faults
    msg_drop_prob=0.1,
    msg_duplicate_prob=0.0,
    msg_reorder_prob=0.0,
    base_latency_steps=1,
    jitter_steps=0,
    ttl_steps=None,
    bandwidth_bits=None,
    max_msg_dims=None,

    # topology / adversarial channel effects
    partitioned_agents_fn=None,
    jammed_agents_fn=None,
    jam_drop_prob=1.0,
    spoof_prob=0.0,
    spoof_scale=1.0,
    replay_prob=0.0,

    # agent-local faults
    crash_prob=0.0,
    crash_duration=5,
    sensor_noise_prob=0.0,
    sensor_noise_scale=0.0,
    actuator_fault_prob=0.0,
    actuator_fault_mode="noop",

    # byzantine behavior
    byzantine_agents=None,
    byzantine_comm_corrupt_prob=0.0,
    byzantine_action_prob=0.0,
)

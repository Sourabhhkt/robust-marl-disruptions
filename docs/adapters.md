
---

## `docs/adapters.md`

```md
# Adapters

Adapters are the benchmark-specific translation layer of the repository.

Their role is to tell the universal wrapper:

- how to interpret actions
- how to interpret messages
- how to interpret observations
- what a safe noop action is
- how to sample a random action

Without adapters, the wrapper would need benchmark-specific branches for every environment.

---

## Why Adapters Exist

Different benchmarks can represent communication very differently.

Examples:

- Some benchmarks may include communication explicitly in the action:
  - `(env_action, msg_out)`

- Some benchmarks may represent communication implicitly:
  - the last `k` entries of the observation
  - a dictionary field inside the observation
  - some benchmark-specific latent state

- Some benchmarks may use simple integer noop actions
- Others may require a specific action object as noop

A universal wrapper cannot infer these semantics by itself. The adapter provides that mapping.

---

## Core Idea

The adapter tells the wrapper:

> “For this benchmark, here is what counts as the real action, the message, the observation, and the safe fallback behavior.”

The wrapper then uses those answers while the core fault/channel machinery stays benchmark-agnostic.

---

## Main Adapter Responsibilities

A benchmark adapter should typically define the following functionality.

### 1. `split_action(agent, action)`
This separates the input policy action into:
- the actual environment action
- the explicit outgoing message, if any

Examples:
- if the action is already a plain env action, return `(action, None)`
- if the action is `(env_action, msg_out)`, return those two pieces separately

### 2. `message_dim(env, agent, obs=None)`
Returns the size of the communication payload for this benchmark.

Examples:
- fixed message length
- length of an observation slice
- zero if no communication is used

### 3. `extract_outgoing_message(env, agent, obs, env_action, explicit_msg_out)`
Returns the message that should be sent from this agent.

Examples:
- use the explicit message from the action
- derive the message from part of the observation
- derive the message from some benchmark-specific control state

### 4. `inject_incoming_message(env, agent, obs, msg_in)`
Writes the delivered message back into the observation in a benchmark-specific way.

Examples:
- concatenate it to the observation vector
- place it in a dictionary field
- overwrite an observation slice

### 5. `noop_action(env, agent)`
Returns a safe fallback action for this benchmark.

This is especially important under:
- crash faults
- actuator faults
- noop-mode fault behavior

### 6. `random_action(env, agent)`
Returns a random valid action for the benchmark.

This is especially important under:
- random actuator faults
- byzantine action corruption

---

## Helper Functions

Some helper utilities can reduce adapter boilerplate.

### `split_action_augmented(action)`
Useful when actions may be:
- just `env_action`
- or `(env_action, msg_out)`

This helper tries to split them consistently.

### `inject_msg_into_obs(obs, msg_in, mode=..., key=...)`
Useful when received communication should be injected into the observation in a common format.

Examples of injection modes:
- concatenate
- dictionary field
- tuple form

These helpers are convenient for explicit-communication settings.

---

## Universal Adapter Base

A universal adapter base class typically provides the common interface:

- `split_action`
- `message_dim`
- `extract_outgoing_message`
- `inject_incoming_message`
- `noop_action`
- `random_action`

This base class can be subclassed for benchmark-specific behavior.

Its purpose is not to implement every benchmark directly, but to define the common contract expected by the wrapper.

---

## Explicit Communication Benchmarks

In some benchmarks, communication is represented directly in the action.

Example:

```python
action = (env_action, msg_out)

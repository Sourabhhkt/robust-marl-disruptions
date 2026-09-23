# Ecosystem integration (Milestone 10, WP5)

The proposal names curated external repositories (OWL, AgentNet, TapeAgents) to
be integrated for reproducible testing. This page documents how each maps onto
the suite.

## AgentNet (agent-based graph neural networks; Martinkus et al., ICLR 2023)

AgentNet's core idea is agents that *walk a graph and aggregate neighborhood
information through learned attention*. The suite's learned coordinators are an
AgentNet-style instantiation specialized to contested communication:

| AgentNet concept            | Suite realization                                  |
|-----------------------------|----------------------------------------------------|
| agent state on a node       | agent's coordination state `x_i`                   |
| neighborhood readout        | delivered (impaired) neighbor messages             |
| learned attention over edges| `coordinators.AttnAggregatorNet` (M9) scores each message by deviation/magnitude |
| node features               | + per-neighbor link quality (age, reliability, precision) in `coordinators_m10.HeteroAttnNet` (M10) |
| sublinear/local computation | aggregation is strictly local to the delivered neighborhood |

The differences are deliberate: the suite's networks are small, dimension-
agnostic, and evaluated under a disruption channel, so the architecture is the
bridge, not a dependency. No AgentNet code is vendored; the mapping above is the
integration contract for anyone bringing a full AgentNet model to the harness
(implement `aggregator(own, vals)` or the `needs_quality` variant and register it
in `coordinators.get_strategy`).

## OWL / TapeAgents / Magentic-One (LLM multi-agent frameworks)

These frameworks coordinate LLM agents over an in-process text message bus. The
adapter `contrib/llm_comm_adapter.py` brings the suite's impairment families to
any such bus:

- `DisruptedTextBus` realizes drop / latency+jitter / token-budget truncation /
  spoofing / Byzantine senders for text messages, seeded and reproducible.
- The mapping to `core.NetworkFaultConfig` semantics is documented in the module
  docstring (bandwidth -> token budget, spoof -> counterfeit message, etc.).
- `demo_reference_game` is the worked demonstrator: a cooperative reference game
  whose task accuracy degrades gracefully with message drop and spoofing and
  collapses under token truncation and latency (the text analogue of the
  rate-distortion knee and staleness cliffs measured on the numeric benchmarks).
  It runs with deterministic mock agents (no API needed); pass `speaker_fn` /
  `listener_fn` callables to substitute real LLM agents (e.g. a Claude API
  wrapper) without changing the harness or metrics.

To stress-test an OWL or TapeAgents pipeline: route its inter-agent messages
through `DisruptedTextBus.send/deliver`, advance `tick()` once per orchestration
round, and read `bus.stats` alongside the task metric.

## Adding further frameworks

`contrib/adapter_experimental.py` (Milestone 8) retains UNVALIDATED scaffolding
for SMAC, Flatland, Overcooked, MAPF, RoboCup, and FRODO; the patterns there and
in the two integrations above (numeric aggregator interface; text bus interface)
cover the two coordination modalities the suite supports.

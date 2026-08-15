# Constraint-Aware MAS/DPS Benchmark & Stress-Test Suite

A benchmark-agnostic framework for evaluating multi-agent systems (MAS) and
distributed problem-solving (DPS) algorithms under communication disruptions,
agent faults, and adversarial interference, with control-theoretic metrics
(Lyapunov stability, consensus convergence rate, information-theoretic
rate-distortion).

An earlier version provided the wrapper-based augmentation framework; this
version adds a hardened harness, communication-native synthetic benchmarks,
baseline algorithms, and the empirical + control-theoretic stress-test study.
Written reports are maintained separately and are not part of this public
code release.

## Layered design

| Layer | File | Role |
|-------|------|------|
| Disruption engine | core.py | NetworkFaultConfig, NetworkChannel (drop/latency/jitter/bandwidth/spoof/jam/partition/replay/TTL + per-source recv_all), FaultModel (crash/sensor-noise/actuator/Byzantine) |
| Translation | adapter.py | Benchmark-specific message/observation/action mapping (ObsSliceCommAdapter, DictMAAdapter) |
| Normalization | env_shims.py | WNTRParallelEnv (dict-based parallel API) |
| Orchestration | wrapper.py | InstrumentedUniversalCommFaultWrapper - drives env/adapter/faults, sender-anchored comm metrics |
| Comm-native benchmarks | synth_envs.py | consensus / rendezvous / formation / flocking (linear_consensus_rollout), DCOP graph-colouring (dcop_rollout: min-sum, DSA) |
| Algorithms | baselines.py | aggregators (mean/median/trimmed), MPE scripted policies, WNTR controller, task registry |
| Metrics | metrics.py | CommLogger + control-theoretic functions (disagreement/Lyapunov, convergence rate, lambda_2, IAE/ISE, rate-distortion) |
| Experiment runner | runner.py | perturbation sweeps, multi-seed 25/50/75 aggregation, CSV + JSON manifest, multiprocessing |
| Figures/analysis | analysis.py | degradation curves, rate-distortion, Lyapunov trajectories, scalability |
| Experiment driver | run_m8.py | runs the full matrix and regenerates all figures |
| Experimental adapters | contrib/adapter_experimental.py | UNVALIDATED scaffolding (SMAC/Flatland/Overcooked/MAPF/RoboCup/FRODO) |

### Milestone 9 additions (advanced coordination + physical infrastructure)

| Layer | File | Role |
|-------|------|------|
| Advanced coordinators | coordinators.py | oracle, graph-attention GNN aggregator, reward-redistribution RECO policy (torch), hybrid MAS+DPS (clustering + event-triggered messaging); strategy registry |
| Heterogeneous teams | hetero.py | per-agent profiles (gain, link reliability, message precision, role) |
| Coordinated rollout | coordinated.py | consensus rollout supporting all seven strategies + heterogeneity (superset of synth_envs; M8 path untouched) |
| Infrastructure benchmarks | infra_envs.py | distributed economic dispatch by consensus on electric power (pandapower) and gas (pandapipes) networks, plus an abstract sandbox; physics-validated feasibility |
| Learned-model training | train_m9.py | trains GNN (supervised) + RECO (evolution strategies); reward-redistribution sample-efficiency study; saves checkpoints to models/ |
| M9 task registry | baselines_m9.py | `family__strategy` task dispatch (consensus / rendezvous / dispatch) |
| M9 driver | run_m9.py | resilience, cross-domain, scalability, heterogeneity, communication-efficiency experiments + figures |
| M9 figures | analysis_m9.py | strategy-comparison, scalability, heterogeneity, comm-efficiency, sample-efficiency plots |

```bash
pip install pandapower pandapipes                # electric-power + gas network solvers (M9)
python train_m9.py --device cpu                  # train GNN + RECO -> models/*.pt   (~6 min CPU)
python run_m9.py --seeds 10 --infra-seeds 8 --workers 14   # full M9 comparative matrix + figures
python tests/test_m9.py                          # M9 unit tests
```

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # core (numpy<2, pandas, matplotlib, scipy, networkx)
# optional extras (Python 3.9 pins):
pip install pettingzoo gymnasium mpe2            # MPE benchmarks
pip install "wntr==1.3.2" "setuptools<80"        # water-network benchmark
# torch is only needed for planned (not yet released) learned baselines:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```
A fully pinned environment is in requirements-lock.txt.

## Reproduce the results

```bash
python run_m8.py --seeds 10 --workers 12 --outdir results --figdir figures
```
This writes per-episode rows (results/m8_raw.csv), the aggregated summary
(results/m8_summary.csv), a reproducibility manifest (results/m8_manifest.json),
and all figures (figures/).

Run the tests:
```bash
python tests/test_core.py && python tests/test_metrics.py
```

## Quick example

```python
from core import NetworkFaultConfig
from baselines import run_task

cfg = NetworkFaultConfig(seed=0, msg_drop_prob=0.5)
rec = run_task("consensus_mean", cfg, seed=0)
print(rec["final_disagreement"], rec["lyap_rho"], rec["delivery_rate"])
```

See docs/ for the architecture, fault models, adapters, and how to add a new
benchmark.

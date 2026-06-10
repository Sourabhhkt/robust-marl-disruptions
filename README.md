# Constraint-Aware MAS/DPS Benchmark & Stress-Test Suite

A benchmark-agnostic framework for evaluating multi-agent systems (MAS) and
distributed problem-solving (DPS) algorithms under communication disruptions,
agent faults, and adversarial interference, with control-theoretic metrics
(Lyapunov stability, consensus convergence rate, information-theoretic
rate-distortion).

This framework provides a hardened evaluation harness, communication-native
synthetic benchmarks, baseline coordination algorithms, and an empirical plus
control-theoretic stress-test study. Written reports are maintained separately and
are not part of this public code release.

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

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # core (numpy<2, pandas, matplotlib, scipy, networkx)
# optional extras (Python 3.9 pins):
pip install pettingzoo gymnasium mpe2            # MPE benchmarks
pip install "wntr==1.3.2" "setuptools<80"        # water-network benchmark
pip install torch --index-url https://download.pytorch.org/whl/cpu
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

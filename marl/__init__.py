"""Milestone 10 deep cooperative MARL baselines (IPPO / MAPPO / QMIX).

Communication-native cooperative environments (Dec-POMDPs) that run over the same
disruption channel as the rest of the suite, plus compact in-house implementations
of independent PPO, multi-agent PPO (centralized critic), and QMIX, with a shared
trainer CLI. The learned policies are evaluated under the Milestone 8/9 disruption
suite and compared against the classical and advanced coordinators.
"""

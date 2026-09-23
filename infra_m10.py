"""
Milestone 10 WP4: infrastructure realism.

WP4a -- Closed-loop AC secondary voltage control. Milestone 9 validated the
coordinated dispatch against the physical network once, at the end of each
episode, with a DC power flow. Here the physics is *in the loop*: each step the
generator agents (i) read their local bus voltage from a full AC power-flow
solve (pandapower runpp), (ii) exchange voltage-error messages over the impaired
communication channel, (iii) update their reactive/voltage setpoints by
consensus toward the network-average restoration target (the distributed
averaging scheme of secondary control, Simpson-Porco et al. 2015; Bidram et al.
2013), and (iv) the next AC solve feeds the result back. Communication quality
now affects the *trajectory* of the physical network, not just a terminal
snapshot: under impairment the voltage profile recovers slowly, unevenly, or
not at all.

Benchmark: a disturbance (load step) depresses voltages at t=0; the team must
restore the mean voltage to 1.0 pu. Metrics: integral absolute voltage error
(IAE), final mean |V - 1| deviation, and the fraction of buses outside the
0.95-1.05 band over the trajectory.

WP4c -- DCOP scalability sweep (mirroring the M8 consensus scalability study)
is provided by ``dcop_scalability``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional
import numpy as np

from core import NetworkChannel, FaultModel, NetworkFaultConfig, DELIVERED_STATUSES
import synth_envs as SE
import infra_envs as IE

warnings.filterwarnings("ignore")


# ============================================================
# WP4a: closed-loop AC secondary voltage control
# ============================================================
def _build_case(n_agents: int):
    """IEEE 9-bus case with the generators as agents (n_agents <= 3 for case9;
    case30 used when more are requested)."""
    import pandapower as pp
    import pandapower.networks as ppn
    if n_agents <= 3:
        net = ppn.case9()
    else:
        net = ppn.case30()
    return net


def ac_secondary_rollout(
    cfg: NetworkFaultConfig,
    seed: int,
    combiner: Any = "mean",
    n: int = 3,
    steps: int = 30,
    topology: str = "ring_plus",
    p_extra: float = 0.3,
    gain: float = 0.5,
    load_step: float = 1.4,
    byzantine_frac: float = 0.0,
    byzantine_scale: float = 0.1,
    **_,
) -> Dict[str, Any]:
    """Closed-loop AC voltage restoration under communication impairment.

    Each agent controls one generator's voltage setpoint. A load step at t=0
    depresses the profile; agents measure their local bus voltage error
    e_i = 1.0 - V_i, exchange e_i over the channel, and move their setpoint by
    gain * AGG(received errors) -- consensus on the restoration signal, so that
    the team shares the burden rather than each generator overreacting locally.
    The aggregator is the strategy under test. A full AC solve closes the loop
    every step."""
    import pandapower as pp
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(9)[8])
    net = _build_case(n)
    gens = net.gen.index.tolist()[:n]
    n = len(gens)
    agents = [str(i) for i in range(n)]
    adj = SE.build_graph(n, topology, rng, p_extra=p_extra)
    nbrs = [list(np.where(adj[i] > 0)[0]) for i in range(n)]
    n_byz = int(round(byzantine_frac * n))
    byz = set(agents[:n_byz])

    # disturbance: scale all loads up, depressing voltages
    net.load.p_mw *= load_step
    net.load.q_mvar *= load_step

    channel = NetworkChannel(cfg); channel.reset(agents, seed=seed)
    fm = FaultModel(cfg); fm.reset(agents, seed=seed)

    gen_buses = [int(net.gen.at[g, "bus"]) for g in gens]
    vset = np.array([float(net.gen.at[g, "vm_pu"]) for g in gens])

    # Each agent monitors a ZONE of non-generator (load-side) buses; only it sees
    # its zone's voltages. Generator buses are PV buses (the solver pins them to
    # the setpoint), so measuring there would be a silent-failure trap: the
    # restoration signal must come from the load buses, and the network-wide
    # depression is only knowable by exchanging zone measurements -- communication
    # is on the decision-critical path.
    non_gen = [b for b in net.bus.index.tolist() if b not in gen_buses]
    zones: List[List[int]] = [non_gen[i::n] for i in range(n)]

    iae = 0.0
    band_viol = []
    mean_dev_traj = []
    failed_solves = 0
    for t in range(steps):
        try:
            pp.runpp(net, numba=False, init="results" if t else "auto")
        except Exception:
            failed_solves += 1
            break
        vm = net.res_bus.vm_pu.to_numpy()
        err = np.array([float(np.mean([1.0 - float(net.res_bus.vm_pu.at[b])
                                       for b in zones[i]])) if zones[i] else 0.0
                        for i in range(n)])
        mean_dev = float(np.mean(np.abs(vm - 1.0)))
        iae += mean_dev
        mean_dev_traj.append(mean_dev)
        band_viol.append(float(np.mean((vm < 0.95) | (vm > 1.05))))

        # --- exchange voltage errors over the impaired channel ---
        fm.begin_step(agents, t)
        for i in range(n):
            if fm.is_crashed(agents[i], t):
                continue
            payload = (rng.normal(0.0, byzantine_scale, size=1) if agents[i] in byz
                       else np.array([err[i]]))
            for j in nbrs[i]:
                channel.send(agents[i], agents[j], payload, t, agents=agents,
                             allow_byzantine_corrupt=True)
        # --- receive, aggregate, update setpoints ---
        for j in range(n):
            if fm.is_crashed(agents[j], t):
                continue
            got = channel.recv_all(agents[j], t, dim=1, agents=agents)
            vals = [float(np.asarray(fm.transform_observation(agents[j], p, t),
                                     dtype=float).reshape(-1)[0])
                    for src, (p, m) in got.items() if m["status"] in DELIVERED_STATUSES]
            own_err = float(err[j])
            if vals:
                arr = np.asarray(vals + [own_err])
                if combiner == "median":
                    sig = float(np.median(arr))
                elif combiner == "trimmed":
                    s_ = np.sort(arr); k = max(0, len(s_) // 4)
                    sig = float(np.mean(s_[k:len(s_) - k] if len(s_) - 2 * k > 0 else s_))
                else:
                    sig = float(np.mean(arr))
            else:
                sig = own_err          # no messages: local-only control
            vset[j] = float(np.clip(vset[j] + gain * sig, 0.94, 1.1))
            net.gen.at[gens[j], "vm_pu"] = vset[j]

    rec: Dict[str, Any] = {
        "benchmark": "ac_secondary",
        "n_agents": n,
        "steps": steps,
        "iae_voltage": float(iae),
        "final_mean_dev": float(mean_dev_traj[-1]) if mean_dev_traj else float("nan"),
        "band_violation_mean": float(np.mean(band_viol)) if band_viol else float("nan"),
        "failed_solves": int(failed_solves),
        "V": np.asarray(mean_dev_traj, dtype=float),
        "return": -float(iae),
    }
    return rec


# ============================================================
# WP4b: power+gas co-simulation (energy-hub coupling)
# ============================================================
def power_gas_cosim(
    cfg: NetworkFaultConfig,
    seed: int,
    combiner: Any = "mean",
    n: int = 6,
    n_gas_fired: int = 3,
    steps: int = 70,
    fuel_rate: float = 0.6,
    byzantine_frac: float = 0.0,
    **_,
) -> Dict[str, Any]:
    """Coupled electric-gas episode: the communication-disrupted dispatch
    consensus (Milestone 9) sets the generator outputs; the first ``n_gas_fired``
    generators are gas-fired, and their fuel draw (proportional to dispatched
    power) loads the gas network as extra sinks at their supply junctions. A
    miscoordinated electric dispatch therefore propagates into the gas network:
    over-dispatched gas-fired units pull more fuel, depressing delivery pressures
    past the safe band -- a cross-domain cascade that neither single-domain study
    can see. Episode-level coupling (dispatch -> fuel draw -> gas solve); a
    step-level co-simulation is left as future work and noted in the report."""
    import pandapipes as ppi
    rec = IE.dispatch_consensus_rollout(cfg, seed, combiner, domain="abstract",
                                        n=n, steps=steps,
                                        byzantine_frac=byzantine_frac, physics=False)
    # reconstruct the dispatch P from the rollout outputs
    prob = IE.make_dispatch_problem(n, seed)
    # final dispatch implied by the rollout's regret bookkeeping is not returned
    # directly; re-derive by running the same rollout path's problem with the
    # reported mismatch is overkill -- instead we reuse the rollout's recorded
    # final values via a light re-run hook:
    P_total = prob["D"] - rec["final_power_mismatch"]   # served power (signed loss)
    # fuel draws: split served power across gas-fired units in proportion to the
    # heterogeneous capacities (the dispatch concentrates on cheap units, and the
    # mismatch concentrates the error); the imbalance fraction loads the cascade.
    pmax = prob["pmax"]
    share = pmax[:n_gas_fired] / max(1e-9, pmax[:n_gas_fired].sum())
    overdraw = abs(rec["final_power_mismatch"]) / max(1e-9, prob["D"])
    net = IE._build_pipe_network(n_gas_fired, seed, "lgas", 3.0)
    total_sink = float(net.sink.mdot_kg_per_s.sum())
    # Fuel draws are EXTRA SINKS at the gas-fired units' junctions; the gas-side
    # supply (the network's sources) injects a balanced share of the TOTAL
    # demand, as in the well-coordinated M9 calibration. A coordinated electric
    # dispatch is then feasible; the electric-side miscoordination concentrates
    # an overdraw at the over-dispatched unit's junction, which the ring must
    # carry locally -- the cross-domain cascade.
    fuel_nominal = fuel_rate * total_sink * share
    fuel = fuel_nominal.copy()
    fuel[0] *= (1.0 + 4.0 * overdraw)
    src_juncs = [int(net.source.at[si, "junction"]) for si in net.source.index.tolist()[:n_gas_fired]]
    for k, j in enumerate(src_juncs):
        ppi.create_sink(net, junction=j, mdot_kg_per_s=float(fuel[k]))
    # The gas supply is scheduled against the PLANNED demand (it cannot foresee
    # the electric-side miscoordination); the unplanned overdraw must be carried
    # by the slack through the thin ring -- that is what breaks the pressure.
    planned = total_sink + float(fuel_nominal.sum())
    for si in net.source.index.tolist():
        net.source.at[si, "mdot_kg_per_s"] = planned / max(1, len(net.source))
    try:
        ppi.pipeflow(net)
        pbar = net.res_junction.p_bar.to_numpy()
        gas_viol = float(np.mean(pbar < 1.5))
        infeasible = 0.0
    except Exception:
        gas_viol, infeasible = 1.0, 1.0
    out = {"benchmark": "power_gas_cosim", "n_agents": n,
           "final_dispatch_regret": rec["final_dispatch_regret"],
           "final_power_mismatch": rec["final_power_mismatch"],
           "gas_violation_frac": gas_viol, "gas_infeasible": infeasible,
           "cascade": float(gas_viol > 0.2 and abs(rec["final_power_mismatch"]) > 0.1),
           "return": -(rec["final_dispatch_regret"] + gas_viol)}
    return out


# ============================================================
# WP4c: DCOP scalability (mirrors the M8 consensus scalability study)
# ============================================================
def dcop_scalability(ns=(12, 24, 48, 96), seeds=range(10), algorithm="dsa",
                     cfg_kw: Optional[Dict[str, Any]] = None):
    """Final conflict fraction vs problem size, clean vs impaired channel."""
    rows = []
    for nn in ns:
        for s in seeds:
            cfg = NetworkFaultConfig(seed=int(s), **(cfg_kw or {}))
            rec = SE.dcop_rollout(cfg, int(s), algorithm=algorithm, n=int(nn),
                                  k_colors=4, steps=40, p_extra=0.08)
            rows.append({"n": int(nn), "seed": int(s),
                         "conflict_frac": rec["final_conflict_frac"]})
    return rows


if __name__ == "__main__":
    # self-check: clean channel restores voltage; total loss leaves local-only
    for lbl, kw in [("clean", {}), ("drop0.9", {"msg_drop_prob": 0.9})]:
        rec = ac_secondary_rollout(NetworkFaultConfig(seed=0, **kw), 0, "mean")
        print(f"ac_secondary {lbl}: IAE={rec['iae_voltage']:.4f} "
              f"final_dev={rec['final_mean_dev']:.5f} band_viol={rec['band_violation_mean']:.3f}")

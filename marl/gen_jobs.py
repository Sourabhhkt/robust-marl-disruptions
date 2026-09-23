"""Generate the WP1 training job manifest for run_local.py.

Synthetic continuous tasks use the continuous-action methods (ippo/mappo/maddpg);
QMIX is reserved for the discrete MPE tasks (added separately) per the M9
discrete-action caveat. Single-threaded jobs (1 core) so a dozen run in parallel.
"""
import argparse
VENV_PY = "/home/nvsai/darpa/.venv/bin/python3"

CONT_ALGOS = ["ippo", "mappo", "maddpg"]   # continuous methods -> synthetic tasks
SYNTH_ENVS = ["consensus", "dispatch"]
DISC_ALGOS = ["ippo", "mappo", "qmix"]     # discrete methods -> MPE tasks
MPE_ENVS = ["mpe_spread", "mpe_reference"]
REGIMES = ["clean", "disrupt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=60000, help="synthetic-task steps")
    ap.add_argument("--mpe-steps", type=int, default=400000, help="MPE-task steps (harder)")
    ap.add_argument("--include-mpe", action="store_true")
    ap.add_argument("--out", default="marl/jobs_wp1.txt")
    args = ap.parse_args()
    lines = ["# WP1 training manifest. 1 core each (single-threaded via the scheduler).",
             "# run: python ../m10_plan/compute/run_local.py --jobs jobs_wp1.txt --max-cores 14"]
    n = 0
    for algo in CONT_ALGOS:
        for env in SYNTH_ENVS:
            for regime in REGIMES:
                for s in range(args.seeds):
                    lines.append(f"{VENV_PY} -m marl.train --algo {algo} --env {env} "
                                 f"--regime {regime} --seed {s} --steps {args.steps} "
                                 f"--outdir marl/checkpoints")
                    n += 1
    if args.include_mpe:
        lines.append("# --- MPE discrete tasks (QMIX belongs here) ---")
        for algo in DISC_ALGOS:
            for env in MPE_ENVS:
                for regime in REGIMES:
                    for s in range(args.seeds):
                        lines.append(f"{VENV_PY} -m marl.train --algo {algo} --env {env} "
                                     f"--regime {regime} --seed {s} --steps {args.mpe_steps} "
                                     f"--outdir marl/checkpoints")
                        n += 1
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {n} jobs to {args.out}")


if __name__ == "__main__":
    main()

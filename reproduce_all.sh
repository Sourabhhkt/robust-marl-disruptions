#!/usr/bin/env bash
# Reproduce every released result of the suite (Milestones 8, 9, 10) from a
# single command, deterministically at the released seeds. Stages can be run
# selectively: ./reproduce_all.sh [m8|m9|m10-train|m10-eval|m10-extras|all]
#
# Environment: Python 3.9 venv with the pinned packages (see requirements.txt;
# pip install -e ".[all]" gives the full set). The learned-model checkpoints
# shipped under models/ and marl/checkpoints/ are what the reports use --
# retraining is optional (train stages) and evaluation never requires it.
set -euo pipefail
PY="${PY:-python3}"
STAGE="${1:-all}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run_m8() {
  echo "== M8: baseline stress-test matrix + figures =="
  $PY run_m8.py --seeds 10 --workers 12 --outdir results --figdir figures
}

run_m9() {
  echo "== M9: comparative strategy evaluation =="
  # optional retrain: $PY train_m9.py --device cpu
  $PY run_m9.py --seeds 10 --infra-seeds 20 --workers 12 \
      --outdir results_m9 --figdir figures_m9
}

run_m10_train() {
  echo "== M10: deep-MARL training matrices (synthetic + MPE) =="
  $PY -m marl.gen_jobs --seeds 5 --steps 50000 --include-mpe --out marl/jobs_wp1.txt
  $PY ../m10_plan/compute/run_local.py --jobs marl/jobs_wp1.txt --max-cores 14 \
      --logdir marl/joblogs || \
  $PY m10_plan/compute/run_local.py --jobs marl/jobs_wp1.txt --max-cores 14 \
      --logdir marl/joblogs
}

run_m10_eval() {
  echo "== M10: learned-vs-classical evaluation + figures =="
  for spec in "consensus disrupt" "consensus clean" "dispatch disrupt"; do
    set -- $spec
    $PY -m marl.run_eval --env $1 --regime $2 --eval-seeds 10 \
        --outdir results_m10 --figdir figures_m10
  done
  $PY cross_modality.py --out results_m10/wp6_transfer.json
}

run_m10_extras() {
  echo "== M10: WP2/WP3/WP4/WP5 studies =="
  $PY train_m10.py --epochs 4000                       # heterogeneity-aware GNN
  $PY attack_m10.py --frac 0.25 --seeds 6              # worst-case attack search
  $PY combined_perturbations.py --seeds 8              # paired impairments
  $PY contrib/llm_comm_adapter.py --episodes 300       # LLM-coordination demo
  $PY - <<'PYEOF'
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, infra_m10 as IM
from core import NetworkFaultConfig
rows=[]
for lbl,kw,byz in [("clean",{},0.0),("drop0.5",{"msg_drop_prob":0.5},0.0),
                   ("drop0.9",{"msg_drop_prob":0.9},0.0),
                   ("byz0.33",{"byzantine_comm_corrupt_prob":1.0,"spoof_scale":2.0},0.33)]:
    for s in range(10):
        r=IM.ac_secondary_rollout(NetworkFaultConfig(seed=s,**kw),s,"mean",byzantine_frac=byz)
        rows.append({"cond":lbl,"seed":s,"iae":r["iae_voltage"],
                     "final_dev":r["final_mean_dev"],"band_viol":r["band_violation_mean"]})
    for s in range(10):
        r=IM.power_gas_cosim(NetworkFaultConfig(seed=s,**kw),s,"mean",byzantine_frac=byz)
        rows.append({"cond":lbl+"_cosim","seed":s,"gas_viol":r["gas_violation_frac"],
                     "cascade":r["cascade"]})
pd.DataFrame(rows).to_csv("results_m10/wp4_infra.csv",index=False)
print("wrote results_m10/wp4_infra.csv")
PYEOF
}

case "$STAGE" in
  m8) run_m8 ;;
  m9) run_m9 ;;
  m10-train) run_m10_train ;;
  m10-eval) run_m10_eval ;;
  m10-extras) run_m10_extras ;;
  all) run_m8; run_m9; run_m10_train; run_m10_eval; run_m10_extras ;;
  *) echo "usage: $0 [m8|m9|m10-train|m10-eval|m10-extras|all]"; exit 1 ;;
esac
echo "reproduce stage '$STAGE' complete."

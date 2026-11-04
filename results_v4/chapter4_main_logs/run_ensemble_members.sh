#!/bin/bash
# Ensemble members for A6-Quantile: seeds 1-4 on FD002/FD004 (seed 0 already
# trained in chapter4_main). split_seed stays 0 so all members share the
# registered engine split. 2-way GPU parallelism.
set -u
cd /home/v2/.worktrees/ch4-hcq-cqr/code
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/.worktrees/ch4-hcq-cqr/code
LOG_DIR=results_v4/chapter4_main_logs/ensemble
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"; shift
  echo "[$(date '+%F %T')] START $tag" >> "$LOG_DIR/progress.log"
  bash -c "$*" > "$LOG_DIR/${tag}.log" 2>&1
  echo "[$(date '+%F %T')] END $tag rc=$?" >> "$LOG_DIR/progress.log"
}

for ds in FD002 FD004; do
  for seed in 1 2 3 4; do
    cmd="$PY -m rul_fusion.quantile_cli --dataset $ds --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset --history 50 --rul_cap 125 --split_seed 0 --seed $seed --output_dir results_v4/chapter4_main/quantile_cqr --calibration_mode global --alpha 0.2 --epochs 41 --batch_size 100 --device auto"
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 10; done
    run_one "${ds}_seed${seed}" "$cmd" &
  done
done
wait
echo "[$(date '+%F %T')] ALL MEMBERS DONE" >> "$LOG_DIR/progress.log"

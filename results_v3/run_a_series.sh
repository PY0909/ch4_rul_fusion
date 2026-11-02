#!/usr/bin/env bash
# A1-A7 single-seed screening runs per docs/plans/2026-10-29-b3-trend-preserving-redesign.md:
# frozen data protocol (grouped scaler, cap=125 train, raw test) + registered
# fixed_cuts validation protocol + FCSTGNN recipe (Adam, wd 0, batch 100,
# 41 epochs, no scheduler/clip). Output layout: results_v3/fixed_cuts/<arm>/.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V3=/home/v2/code/results_v3
LOG_DIR=$V3/logs
mkdir -p "$LOG_DIR"
RECIPE="--optimizer adam --weight_decay 0 --batch_size 100 --epochs 41 --patience 41 --no_scheduler --no_grad_clip"

for ARM in A1 A2 A3 A4 A5 A6 A7; do
  for DS in FD002 FD004; do
    RUN_DIR=$V3/fixed_cuts/$ARM/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] $ARM $DS"; continue; fi
    echo "[$(date '+%F %T')] start $ARM $DS"
    $PY -m rul_fusion.cli \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap 125 --scaler grouped \
      --encoder patch_cnn --variant "$ARM" \
      --validation_protocol fixed_cuts --seed 0 \
      $RECIPE \
      --output_dir "$V3/fixed_cuts/$ARM" \
      > "$LOG_DIR/${ARM}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end $ARM $DS exit=$?"
  done
done
echo "A-SERIES SCREENING FINISHED"

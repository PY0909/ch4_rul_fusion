#!/usr/bin/env bash
# 1) FCSTGNN reference under the frozen protocol (FD002, FD004).
# 2) Cycle-in-conditions ablation for B2-B5 (vs existing settings-only runs in results_v2/ablation).
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V2=/home/v2/code/results_v2
LOG_DIR=$V2/logs
mkdir -p "$LOG_DIR"

for DS in FD002 FD004; do
  if [ -f "$V2/fcstgnn/$DS/seed_0/metrics.json" ]; then echo "[skip] fcstgnn $DS"; continue; fi
  echo "[$(date '+%F %T')] start FCSTGNN $DS"
  $PY /home/v2/code/scripts/run_fcstgnn.py \
    --dataset "$DS" --data_root "$DATA_ROOT" --seed 0 \
    --output_dir "$V2/fcstgnn" \
    > "$LOG_DIR/fcstgnn_${DS}.log" 2>&1
  echo "[$(date '+%F %T')] end FCSTGNN $DS exit=$?"
done

declare -A FLAGS
FLAGS[B2]="--num_scales 3 --disable_graph --disable_shared_specific"
FLAGS[B3]="--num_scales 3 --disable_shared_specific"
FLAGS[B4]="--num_scales 3"
FLAGS[B5]="--num_scales 3 --alpha_shared 0.01 --alpha_specific 0.01"

for BLOCK in B2 B3 B4 B5; do
  for DS in FD002 FD004; do
    RUN_DIR=$V2/cycle_ablation/$BLOCK/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] cycle $BLOCK $DS"; continue; fi
    echo "[$(date '+%F %T')] start cycle $BLOCK $DS"
    $PY -m rul_fusion.cli \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap 125 --scaler grouped --seed 0 \
      --condition_features settings_cycle \
      ${FLAGS[$BLOCK]} \
      --output_dir "$V2/cycle_ablation/$BLOCK" \
      > "$LOG_DIR/cycle_${BLOCK}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end cycle $BLOCK $DS exit=$?"
  done
done
echo "ALL FINISHED"

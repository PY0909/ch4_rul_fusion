#!/usr/bin/env bash
# A6 hyperparameter screening on N-CMAPSS (DS03, DS08a), seed=0.
# Selection criterion: val_fc (fixed_cuts endpoint RMSE on dev val units).
# Axes: capacity (d_model/routers/heads), weight decay, dropout,
# condition mode (full/mean/none), lr+budget. Grid reported in full.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V3=/home/v2/code/results_v3
OUT=$V3/ncmapss_tune
LOG_DIR=$V3/logs
mkdir -p "$LOG_DIR"

declare -A CFG
CFG[t0_default]="--d_model 64 --num_routers 4 --num_heads 4"
CFG[t1_d32]="--d_model 32 --num_routers 2 --num_heads 2"
CFG[t2_d16]="--d_model 16 --num_routers 2 --num_heads 2"
CFG[t3_d32_wd]="--d_model 32 --num_routers 2 --num_heads 2 --weight_decay 1e-3"
CFG[t4_d32_do02]="--d_model 32 --num_routers 2 --num_heads 2 --dropout 0.2"
CFG[t5_d32_mean]="--d_model 32 --num_routers 2 --num_heads 2 --condition_mode mean"
CFG[t6_d32_none]="--d_model 32 --num_routers 2 --num_heads 2 --condition_mode none"
CFG[t7_d32_lr3e4]="--d_model 32 --num_routers 2 --num_heads 2 --lr 3e-4 --epochs 600"

for CFG_NAME in t0_default t1_d32 t2_d16 t3_d32_wd t4_d32_do02 t5_d32_mean t6_d32_none t7_d32_lr3e4; do
  for DS in DS03 DS08a; do
    RUN_DIR=$OUT/$CFG_NAME/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] $CFG_NAME $DS"; continue; fi
    echo "[$(date '+%F %T')] start $CFG_NAME $DS"
    $PY /home/v2/code/scripts/run_ncmapss.py \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --model A6 --seed 0 \
      --output_dir "$OUT/$CFG_NAME" \
      ${CFG[$CFG_NAME]} \
      > "$LOG_DIR/tune_${CFG_NAME}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end $CFG_NAME $DS exit=$?"
  done
done
echo "A6 TUNING FINISHED"

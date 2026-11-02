#!/usr/bin/env bash
# B0->B5 re-run under frozen protocol: rul_cap=125 (train side), grouped scaler,
# endpoint-val checkpoint selection, test evaluated once on raw RUL. seed=0.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
OUT_ROOT=/home/v2/code/results_v2/ablation
LOG_DIR=$OUT_ROOT/logs
mkdir -p "$LOG_DIR"

declare -A FLAGS
FLAGS[B0]="--num_scales 1 --disable_router --disable_graph --disable_shared_specific"
FLAGS[B1]="--num_scales 3 --disable_router --disable_graph --disable_shared_specific"
FLAGS[B2]="--num_scales 3 --disable_graph --disable_shared_specific"
FLAGS[B3]="--num_scales 3 --disable_shared_specific"
FLAGS[B4]="--num_scales 3"
FLAGS[B5]="--num_scales 3 --alpha_shared 0.01 --alpha_specific 0.01"

for BLOCK in B0 B1 B2 B3 B4 B5; do
  for DS in FD002 FD004; do
    RUN_DIR=$OUT_ROOT/$BLOCK/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] $BLOCK $DS"; continue; fi
    echo "[$(date '+%F %T')] start $BLOCK $DS"
    $PY -m rul_fusion.cli \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap 125 --scaler grouped --seed 0 \
      ${FLAGS[$BLOCK]} \
      --output_dir "$OUT_ROOT/$BLOCK" \
      > "$LOG_DIR/${BLOCK}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end $BLOCK $DS exit=$?"
  done
done
echo "ABLATION FINISHED"

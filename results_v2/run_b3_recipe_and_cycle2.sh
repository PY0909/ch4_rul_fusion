#!/usr/bin/env bash
# 1) B3 under the FCSTGNN training recipe (Adam, no wd, batch 100, 41 epochs,
#    no scheduler, no grad clip; endpoint-val selection and raw test unchanged).
# 2) Corrected cycle ablation (previous cycle_ablation/ was run with the flag
#    not wired through; those 8 runs are same-config replicates, not cycle arms).
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V2=/home/v2/code/results_v2
LOG_DIR=$V2/logs
mkdir -p "$LOG_DIR"
RECIPE="--optimizer adam --weight_decay 0 --batch_size 100 --epochs 41 --patience 41 --no_scheduler --no_grad_clip"
B3="--num_scales 3 --disable_shared_specific"

for DS in FD002 FD004; do
  RUN_DIR=$V2/b3_fcstgnn_recipe/$DS/seed_0
  if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] b3-recipe $DS"; continue; fi
  echo "[$(date '+%F %T')] start B3-recipe $DS"
  $PY -m rul_fusion.cli \
    --dataset "$DS" --data_root "$DATA_ROOT" \
    --history 50 --rul_cap 125 --scaler grouped --seed 0 \
    $B3 $RECIPE \
    --output_dir "$V2/b3_fcstgnn_recipe" \
    > "$LOG_DIR/b3_recipe_${DS}.log" 2>&1
  echo "[$(date '+%F %T')] end B3-recipe $DS exit=$?"
done

declare -A FLAGS
FLAGS[B2]="--num_scales 3 --disable_graph --disable_shared_specific"
FLAGS[B3]="--num_scales 3 --disable_shared_specific"
FLAGS[B4]="--num_scales 3"
FLAGS[B5]="--num_scales 3 --alpha_shared 0.01 --alpha_specific 0.01"

for BLOCK in B2 B3 B4 B5; do
  for DS in FD002 FD004; do
    RUN_DIR=$V2/cycle_ablation_v2/$BLOCK/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] cycle2 $BLOCK $DS"; continue; fi
    echo "[$(date '+%F %T')] start cycle2 $BLOCK $DS"
    $PY -m rul_fusion.cli \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap 125 --scaler grouped --seed 0 \
      --condition_features settings_cycle \
      ${FLAGS[$BLOCK]} \
      --output_dir "$V2/cycle_ablation_v2/$BLOCK" \
      > "$LOG_DIR/cycle2_${BLOCK}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end cycle2 $BLOCK $DS exit=$?"
  done
done
echo "ALL FINISHED"

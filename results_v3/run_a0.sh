#!/usr/bin/env bash
# A0 same-protocol reference runs (legacy readout/downsampling/graph, seed-corrected
# original B3+patch_cnn) under the registered A-series protocol.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V3=/home/v2/code/results_v3
LOG_DIR=$V3/logs
mkdir -p "$LOG_DIR"
RECIPE="--optimizer adam --weight_decay 0 --batch_size 100 --epochs 41 --patience 41 --no_scheduler --no_grad_clip"

for DS in FD002 FD004; do
  RUN_DIR=$V3/fixed_cuts/A0/$DS/seed_0
  if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] A0 $DS"; continue; fi
  echo "[$(date '+%F %T')] start A0 $DS"
  $PY -m rul_fusion.cli \
    --dataset "$DS" --data_root "$DATA_ROOT" \
    --history 50 --rul_cap 125 --scaler grouped \
    --encoder patch_cnn --variant A0 \
    --validation_protocol fixed_cuts --seed 0 \
    $RECIPE \
    --output_dir "$V3/fixed_cuts/A0" \
    > "$LOG_DIR/A0_${DS}.log" 2>&1
  echo "[$(date '+%F %T')] end A0 $DS exit=$?"
done
echo "A0 REFERENCE FINISHED"

#!/usr/bin/env bash
# FCSTGNN reference on FD002/FD004 under the exact A-series protocol
# (fixed_cuts validation, FCSTGNN recipe, cap125+grouped) so the four-dataset
# A6-vs-FCSTGNN comparison is fully matched. seed=0.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V3=/home/v2/code/results_v3
LOG_DIR=$V3/logs
mkdir -p "$LOG_DIR"
RECIPE="--optimizer adam --weight_decay 0 --batch_size 100 --epochs 41 --patience 41 --no_scheduler --no_grad_clip"

for DS in FD002 FD004; do
  RUN_DIR=$V3/fixed_cuts/FCSTGNN_ref/$DS/seed_0
  if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] fcstgnn-ref $DS"; continue; fi
  echo "[$(date '+%F %T')] start FCSTGNN-ref $DS"
  $PY -m rul_fusion.cli \
    --dataset "$DS" --data_root "$DATA_ROOT" \
    --model fc_stgnn \
    --history 50 --rul_cap 125 --scaler grouped \
    --validation_protocol fixed_cuts --seed 0 \
    $RECIPE \
    --output_dir "$V3/fixed_cuts/FCSTGNN_ref" \
    > "$LOG_DIR/fcstgnn_ref_${DS}.log" 2>&1
  echo "[$(date '+%F %T')] end FCSTGNN-ref $DS exit=$?"
done
echo "FCSTGNN REFERENCE (FIXED_CUTS) FINISHED"

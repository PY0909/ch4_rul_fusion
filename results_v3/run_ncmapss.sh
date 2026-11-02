#!/usr/bin/env bash
# A6 vs FCSTGNN on N-CMAPSS (DS03, DS08a), single seed, shared protocol:
# fixed_cuts endpoint validation, no cap, unit-grouped split, Adam budget
# registered for N-CMAPSS screening (batch 32, 300 epochs).
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V3=/home/v2/code/results_v3
LOG_DIR=$V3/logs
mkdir -p "$LOG_DIR"

for DS in DS03 DS08a; do
  for MODEL in A6 fc_stgnn; do
    RUN_DIR=$V3/ncmapss/$MODEL/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] $MODEL $DS"; continue; fi
    echo "[$(date '+%F %T')] start $MODEL $DS"
    $PY /home/v2/code/scripts/run_ncmapss.py \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --model "$MODEL" --seed 0 \
      --output_dir "$V3/ncmapss/$MODEL" \
      > "$LOG_DIR/ncmapss_${MODEL}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end $MODEL $DS exit=$?"
  done
done
echo "N-CMAPSS SCREENING FINISHED"

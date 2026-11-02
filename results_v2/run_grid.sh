#!/usr/bin/env bash
# 2x2 protocol grid on B0 config: {rul_cap 0,125} x {scaler global,grouped}, FD002+FD004.
# (raw, global) cell doubles as the post-fix B0 sanity run. Test is always raw RUL.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
OUT_ROOT=/home/v2/code/results_v2/grid
LOG_DIR=/home/v2/code/results_v2/logs
mkdir -p "$LOG_DIR"
B0_FLAGS="--num_scales 1 --disable_router --disable_graph --disable_shared_specific"

for SCALER in global grouped; do
  for CAP in 0 125; do
    for DS in FD002 FD004; do
      RUN_DIR=$OUT_ROOT/${SCALER}_cap${CAP}/$DS/seed_0
      if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] $SCALER cap$CAP $DS"; continue; fi
      echo "[$(date '+%F %T')] start $SCALER cap$CAP $DS"
      $PY -m rul_fusion.cli \
        --dataset "$DS" --data_root "$DATA_ROOT" \
        --history 50 --rul_cap "$CAP" --scaler "$SCALER" --seed 0 \
        $B0_FLAGS \
        --output_dir "$OUT_ROOT/${SCALER}_cap${CAP}" \
        > "$LOG_DIR/${SCALER}_cap${CAP}_${DS}.log" 2>&1
      echo "[$(date '+%F %T')] end $SCALER cap$CAP $DS exit=$?"
    done
  done
done
echo "GRID FINISHED"

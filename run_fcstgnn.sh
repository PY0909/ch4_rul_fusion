#!/usr/bin/env bash
# Unified-protocol FC-STGNN baseline on FD002/FD004.
#
# The baseline is trained by the same runner as the fusion model: engine-grouped
# 80/20 split (split_seed=0), train-only scaler, validation-based checkpoint
# selection, single frozen test evaluation. The original FC-STGNN protocol
# (window-level folds, per-condition MinMax, label/125, best-epoch by test RMSE)
# is deliberately NOT used.
#
# This script is NOT scheduled anywhere; run it manually when GPU time allows.
#
# Env overrides:
#   SEEDS    model seeds                       (default "0 1 2")
#   RUL_CAP  125 = literature-comparable arm   (default 125)
#            0   = raw-label arm, same as the fusion pilot
#   SCALER   grouped | global                  (default grouped)
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
SEEDS=${SEEDS:-"0 1 2"}
RUL_CAP=${RUL_CAP:-125}
SCALER=${SCALER:-grouped}
OUT_ROOT=${OUT_ROOT:-/home/v2/code/results_baselines/FCSTGNN/cap${RUL_CAP}_${SCALER}}
LOG_DIR=$(dirname "$OUT_ROOT")/logs
mkdir -p "$LOG_DIR"

for DS in FD002 FD004; do
  for SEED in $SEEDS; do
    RUN_DIR=$OUT_ROOT/$DS/seed_$SEED
    if [ -f "$RUN_DIR/metrics.json" ]; then
      echo "[skip] FCSTGNN $DS seed $SEED already done"
      continue
    fi
    echo "[$(date '+%F %T')] start FCSTGNN $DS seed $SEED -> $RUN_DIR"
    $PY -m rul_fusion.cli \
      --model fc_stgnn \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap "$RUL_CAP" --scaler "$SCALER" \
      --split_seed 0 --seed "$SEED" \
      --output_dir "$OUT_ROOT" \
      > "$LOG_DIR/FCSTGNN_${DS}_seed${SEED}.log" 2>&1
    status=$?
    echo "[$(date '+%F %T')] end FCSTGNN $DS seed $SEED exit=$status"
    if [ $status -ne 0 ]; then
      echo "FAILED: FCSTGNN $DS seed $SEED" >> "$LOG_DIR/FAILURES.txt"
    fi
  done
done

echo "FCSTGNN RUNS FINISHED"
$PY - <<'EOF'
import json
from pathlib import Path
root = Path("/home/v2/code/results_baselines/FCSTGNN")
for run in sorted(root.glob("cap*/**/seed_*/metrics.json")):
    r = json.loads(run.read_text())
    print(run, f"MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} NASA={r['NASA_Score']:.1f}")
EOF

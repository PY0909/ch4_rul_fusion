#!/usr/bin/env bash
# Single-seed (seed=0) ablation B0->B5 on FD002 and FD004.
# Configs per IMPLEMENTATION_PLAN.md section 5; history=50, raw RUL labels.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
OUT_ROOT=/home/v2/code/results_ablation
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
    if [ -f "$RUN_DIR/metrics.json" ]; then
      echo "[skip] $BLOCK $DS already done"
      continue
    fi
    echo "[$(date '+%F %T')] start $BLOCK $DS -> $RUN_DIR"
    $PY -m rul_fusion.cli \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap 0 --seed 0 \
      --output_dir "$OUT_ROOT/$BLOCK" \
      ${FLAGS[$BLOCK]} \
      > "$LOG_DIR/${BLOCK}_${DS}.log" 2>&1
    status=$?
    echo "[$(date '+%F %T')] end   $BLOCK $DS exit=$status"
    if [ $status -ne 0 ]; then
      echo "FAILED: $BLOCK $DS (see $LOG_DIR/${BLOCK}_${DS}.log)" >> "$LOG_DIR/FAILURES.txt"
    fi
  done
done

echo "ALL RUNS FINISHED"
$PY - <<'EOF'
import json
from pathlib import Path
root = Path("/home/v2/code/results_ablation")
for block in ["B0","B1","B2","B3","B4","B5"]:
    for ds in ["FD002","FD004"]:
        m = root / block / ds / "seed_0" / "metrics.json"
        if m.exists():
            r = json.loads(m.read_text())
            print(f"{block} {ds}: MAE={r['MAE']:.2f} RMSE={r['RMSE']:.2f} NASA={r['NASA_Score']:.1f} best_epoch={r['best_epoch']}")
        else:
            print(f"{block} {ds}: MISSING")
EOF

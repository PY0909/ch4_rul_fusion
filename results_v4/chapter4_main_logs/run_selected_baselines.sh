#!/bin/bash
# Run the requested chapter-4 baselines (A6-Point, A6-Quantile+Global CQR, HNN,
# EDL-NIG) on FD002/FD004, seed 0. Commands come from the registered generator
# in scripts/run_chapter4_baselines.py; the Deep Ensemble row is filtered out
# because it is not part of this batch. Two jobs run concurrently on one GPU.
set -u
cd /home/v2/.worktrees/ch4-hcq-cqr/code
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/.worktrees/ch4-hcq-cqr/code
LOG_DIR=results_v4/chapter4_main_logs
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1"; shift
  echo "[$(date '+%F %T')] START $tag" >> "$LOG_DIR/progress.log"
  bash -c "$*" > "$LOG_DIR/${tag}.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] END $tag rc=$rc" >> "$LOG_DIR/progress.log"
}

CMDS=$($PY scripts/run_chapter4_baselines.py --dataset FD002 --seed 0 | grep -v "method ensemble"; \
       $PY scripts/run_chapter4_baselines.py --dataset FD004 --seed 0 | grep -v "method ensemble")

while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  ds=$(echo "$cmd" | sed -n 's/.*--dataset \([A-Za-z0-9]*\).*/\1/p')
  case "$cmd" in
    *"rul_fusion.cli "*) m="A6point" ;;
    *"quantile_cli"*)    m="quantile_cqr" ;;
    *"method hnn"*)      m="hnn" ;;
    *"method edl_nig"*)  m="edl_nig" ;;
    *) echo "[$(date '+%F %T')] SKIP unknown command" >> "$LOG_DIR/progress.log"; continue ;;
  esac
  tag="${ds}_${m}"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 10; done
  run_one "$tag" "$cmd" &
done <<< "$CMDS"
wait
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG_DIR/progress.log"

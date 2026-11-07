#!/usr/bin/env bash
# One-shot descriptive test evaluation of the frozen N-CMAPSS final arm
# (authorized by the user on 2026-11-07, after 18/18 training verification).
# Re-runs the independent verification first and aborts on any failure;
# each evaluator re-checks hashes again and refuses repeat evaluation.
set -euo pipefail
ROOT=/root/autodl-tmp/ch4_rul_fusion
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
SELECTION="$ROOT/results_v4/ncmapss_loso_frozen/selection.json"
DEVICE=cpu
if nvidia-smi >/dev/null 2>&1; then DEVICE=cuda; fi
cd "$ROOT"
echo "=== pre-test verification $(date '+%F %T') ==="
"$PY" scripts/verify_ncmapss_final_training.py
for ds in DS03 DS08a; do
  echo "=== A6-Quantile evaluate $ds ($DEVICE) $(date '+%F %T') ==="
  "$PY" -u scripts/run_ncmapss_loso_final.py --stage evaluate --eval_test --skip_evaluated \
    --dataset "$ds" --data_root "$DATA" --selection_file "$SELECTION" --device "$DEVICE"
done
echo "=== baselines final_eval ($DEVICE) $(date '+%F %T') ==="
env PYTHONPATH="$ROOT" "$PY" -u scripts/run_ncmapss_loso_baselines.py \
  --phase final_eval --eval_test --skip_evaluated --data_root "$DATA" --device "$DEVICE" \
  --fc_source /home/v2/FCSTGNN/FC_STGNN --selection "$SELECTION" \
  --output_dir "$ROOT/results_v4/ncmapss_loso_final_baselines"
echo "=== evaluation done $(date '+%F %T') ==="

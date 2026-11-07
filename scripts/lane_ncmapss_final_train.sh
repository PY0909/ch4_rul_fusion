#!/usr/bin/env bash
# Final-arm TRAINING stage only (docs/ncmapss-loso-selection-freeze-2026-11-07.md):
# A6-Quantile x2 datasets and eight baselines x2 datasets train on the seven
# search engines for exactly their frozen epochs. Nothing here loads test;
# evaluation is a separate, explicitly authorized step after verification.
set -euo pipefail
ROOT=/root/autodl-tmp/ch4_rul_fusion
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
SELECTION="$ROOT/results_v4/ncmapss_loso_frozen/selection.json"
cd "$ROOT"
for ds in DS03 DS08a; do
  echo "=== A6-Quantile train $ds $(date '+%F %T') ==="
  "$PY" -u scripts/run_ncmapss_loso_final.py --stage train --dataset "$ds" \
    --data_root "$DATA" --selection_file "$SELECTION" --device cuda
done
echo "=== baselines final_train $(date '+%F %T') ==="
env PYTHONPATH="$ROOT" "$PY" -u scripts/run_ncmapss_loso_baselines.py \
  --phase final_train --data_root "$DATA" --device cuda \
  --fc_source /home/v2/FCSTGNN/FC_STGNN --selection "$SELECTION" \
  --output_dir "$ROOT/results_v4/ncmapss_loso_final_baselines"
echo "=== training stage done $(date '+%F %T') ==="

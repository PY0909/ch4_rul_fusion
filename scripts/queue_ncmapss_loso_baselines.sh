#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/ch4_rul_fusion
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset

while tmux has-session -t '=nc_loso_a6' 2>/dev/null; do
  sleep 30
done

"$PY" -c 'import json, pathlib; path = pathlib.Path("/root/autodl-tmp/ch4_rul_fusion/results_v4/ncmapss_loso_budget_compare/manifest.json"); data = json.loads(path.read_text()); assert data["status"] == "complete_development_comparison" and data["n_fold_config_runs"] == 28 and data["test_loaded"] is False, "A6 comparison is incomplete"'

exec env PYTHONPATH="$ROOT" "$PY" -u \
  "$ROOT/scripts/run_ncmapss_loso_baselines.py" \
  --phase development --data_root "$DATA" --device cuda \
  --fc_source /home/v2/FCSTGNN/FC_STGNN \
  --output_dir "$ROOT/results_v4/ncmapss_loso_baselines"

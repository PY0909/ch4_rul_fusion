#!/bin/bash
# N-CMAPSS external-baseline lane (B3-B6) for one dataset (DS03 or DS08a).
# Fault-isolated: a failed run is logged and the lane continues.
# Uses the unified NC budget (300 epochs/patience, batch 32, lr 1e-3, wd 1e-3)
# with per-method paper dropout/MC settings resolved by the runner itself.
DS=${1:?usage: lane_ncmapss_external_baselines.sh DS03\|DS08a}
cd /root/autodl-tmp/ch4_rul_fusion || exit 1
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
OUT=results_v4/ncmapss_comparison/probabilistic
mkdir -p logs
: > "logs/ncmapss_${DS}_failures.log"
for m in lstm_prob tcn_transformer_mc msauq bcpt; do
  echo "=== $DS $m start $(date '+%F %T') ==="
  if [ -f "$OUT/$DS/$m/seed_0/metrics.json" ]; then
    echo "skip completed: $DS/$m"
    continue
  fi
  if [ -d "$OUT/$DS/$m/seed_0" ]; then
    rm -rf "$OUT/$DS/$m/seed_0"
  fi
  "$PY" scripts/run_ncmapss_probabilistic.py --dataset "$DS" --method "$m" \
        --data_root "$DATA" --output_dir "$OUT" \
        --seed 0 --split_seed 0 --eval_test \
    || echo "FAILED: $DS $m rc=$? $(date '+%F %T')" >> "logs/ncmapss_${DS}_failures.log"
done
echo "=== $DS lane done $(date '+%F %T') ==="

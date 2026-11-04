#!/bin/bash
# FD004 external-baseline lane (B3-B6), fault-isolated: a failed run is logged
# and the lane continues, so one method (e.g. bcpt) cannot block the others.
cd /root/autodl-tmp/ch4_rul_fusion || exit 1
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
mkdir -p logs
: > logs/fd004_failures.log
for m in lstm_prob tcn_transformer_mc msauq bcpt; do
  echo "=== FD004 $m start $(date '+%F %T') ==="
  if [ -f "results_v4/external_baselines/FD004/$m/seed_0/metrics.json" ]; then
    echo "skip completed: FD004/$m"
    continue
  fi
  if [ -d "results_v4/external_baselines/FD004/$m/seed_0" ]; then
    rm -rf "results_v4/external_baselines/FD004/$m/seed_0"
  fi
  "$PY" -m rul_fusion.probabilistic_cli --dataset FD004 --method "$m" \
        --data_root "$DATA" --output_dir results_v4/external_baselines \
        --seed 0 --split_seed 0 --calibration_ratio 0.5 \
    || echo "FAILED: FD004 $m rc=$? $(date '+%F %T')" >> logs/fd004_failures.log
done
echo "=== FD004 lane done $(date '+%F %T') ==="

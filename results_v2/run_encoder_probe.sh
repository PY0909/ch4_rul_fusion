#!/usr/bin/env bash
# Encoder attribution under the FCSTGNN recipe (Adam, no wd, batch 100, 41 epochs,
# no scheduler, no clip). All arms share the frozen data protocol.
# Arms: {B0, B3} x {linear, bn, patch_cnn} x {FD002, FD004}; B3+linear already
# exists as results_v2/b3_fcstgnn_recipe, so 10 new runs.
set -u
PY=/root/anaconda3/envs/torch23/bin/python
export PYTHONPATH=/home/v2/code
DATA_ROOT=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset
V2=/home/v2/code/results_v2
OUT=$V2/encoder_probe
LOG_DIR=$V2/logs
mkdir -p "$LOG_DIR"
RECIPE="--optimizer adam --weight_decay 0 --batch_size 100 --epochs 41 --patience 41 --no_scheduler --no_grad_clip"
declare -A FLAGS
FLAGS[B0]="--num_scales 1 --disable_router --disable_graph --disable_shared_specific"
FLAGS[B3]="--num_scales 3 --disable_shared_specific"

run () {  # enc block ds extra
  local ENC=$1 BLOCK=$2 DS=$3 EXTRA=$4
  local RUN_DIR=$OUT/$ENC/$BLOCK/$DS/seed_0
  if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] $ENC $BLOCK $DS"; return; fi
  echo "[$(date '+%F %T')] start $ENC $BLOCK $DS"
  $PY -m rul_fusion.cli \
    --dataset "$DS" --data_root "$DATA_ROOT" \
    --history 50 --rul_cap 125 --scaler grouped --seed 0 \
    $EXTRA $RECIPE ${FLAGS[$BLOCK]} \
    --output_dir "$OUT/$ENC/$BLOCK" \
    > "$LOG_DIR/enc_${ENC}_${BLOCK}_${DS}.log" 2>&1
  echo "[$(date '+%F %T')] end $ENC $BLOCK $DS exit=$?"
}

for DS in FD002 FD004; do
  run linear B0 "$DS" ""                              # recipe baseline (missing pair)
  run bn     B0 "$DS" "--input_norm bn"
  run bn     B3 "$DS" "--input_norm bn"
  run patch_cnn B0 "$DS" "--encoder patch_cnn"
  run patch_cnn B3 "$DS" "--encoder patch_cnn"
done
echo "ENCODER PROBE FINISHED"

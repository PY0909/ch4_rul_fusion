#!/usr/bin/env bash
# Extend the encoder probe to B4/B5: {B4, B5} x patch_cnn x {FD002, FD004}
# under the same FCSTGNN recipe as the existing B0/B3 encoder-probe arms.
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
FLAGS[B4]="--num_scales 3"
FLAGS[B5]="--num_scales 3 --alpha_shared 0.01 --alpha_specific 0.01"

for DS in FD002 FD004; do
  for BLOCK in B4 B5; do
    RUN_DIR=$OUT/patch_cnn/$BLOCK/$DS/seed_0
    if [ -f "$RUN_DIR/metrics.json" ]; then echo "[skip] patch_cnn $BLOCK $DS"; continue; fi
    echo "[$(date '+%F %T')] start patch_cnn $BLOCK $DS"
    $PY -m rul_fusion.cli \
      --dataset "$DS" --data_root "$DATA_ROOT" \
      --history 50 --rul_cap 125 --scaler grouped --seed 0 \
      --encoder patch_cnn $RECIPE ${FLAGS[$BLOCK]} \
      --output_dir "$OUT/patch_cnn/$BLOCK" \
      > "$LOG_DIR/enc_patch_cnn_${BLOCK}_${DS}.log" 2>&1
    echo "[$(date '+%F %T')] end patch_cnn $BLOCK $DS exit=$?"
  done
done
echo "B4/B5 PATCH_CNN PROBE FINISHED"

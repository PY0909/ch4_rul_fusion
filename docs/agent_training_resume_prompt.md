# 交给训练 Agent 的续跑 Prompt

你负责继续 HCQ-CQR 第二阶段训练实验。当前工作树是：

`/home/v2/.worktrees/ch4-hcq-cqr`

进入代码目录后使用：

`/root/anaconda3/envs/torch23/bin/python`

## 当前状态

1. 主实验 ratio=0.5 的冻结 A6 Quantile checkpoint 已存在：
   - `results_v4/chapter4_main/quantile_cqr/global/FD002/seed_0/best.pt`
   - `results_v4/chapter4_main/quantile_cqr/global/FD004/seed_0/best.pt`
2. 第一阶段冻结 checkpoint 的原始 CQR 校准扫描已完成，结果在：
   - `results_v4/calibration_stage1/`
   - `results_v4/calibration_stage1_test/`
3. 早停重训尚未完成：
   - `calibration_ratio_earlystop_v1/ratio_0.5/global/FD002/seed_0/best.pt` 只跑到 epoch 5，不能作为完成结果；
   - FD004 ratio=0.5 没有完成产物；
   - ratio=0.6/0.7 没有早停版完整矩阵。
4. `calibration_ratio_retrain/` 中的部分 ratio=0.6/0.7 结果是 41 epoch 上限实验，不是统一的 `max_epochs=100, patience=12` 早停实验。它们可以保留作探索性结果，但不能替代下面的统一重训。
5. 所有正式候选在测试评估前都必须先完成 validation 选择；不要根据测试结果调参。

## 必须重跑的训练

统一训练以下 6 个组合：

`FD002, FD004 × calibration_ratio {0.5, 0.6, 0.7}`

每个组合都从随机初始化开始，不能从中断的 `best.pt` 续训。配置固定为：

- `seed=0`, `split_seed=0`
- A6、`patch_cnn`、`d_model=64`、4 heads、4 routers
- `history=50`, `rul_cap=125`, grouped scaler
- `validation_protocol=fixed_cuts`
- `epochs=100` 为最大轮数
- `patience=12`
- 早停指标 `val_q50_rmse`
- `--defer_test`
- 输出根目录 `results_v4/calibration_ratio_earlystop_v1/ratio_{ratio}`

推荐逐个串行执行，避免 GPU 显存竞争：

```bash
cd /home/v2/.worktrees/ch4-hcq-cqr/code
for dataset in FD002 FD004; do
  for ratio in 0.5 0.6 0.7; do
    /root/anaconda3/envs/torch23/bin/python -m rul_fusion.quantile_cli \
      --dataset "$dataset" \
      --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
      --output_dir "results_v4/calibration_ratio_earlystop_v1/ratio_${ratio}" \
      --seed 0 --split_seed 0 --calibration_ratio "$ratio" \
      --calibration_mode global --epochs 100 --patience 12 \
      --batch_size 100 --device cuda --defer_test
  done
done
```

每个组合完成后检查以下文件均存在：`best.pt`、`history.csv`、`calibration_only.json`、`calibrator.json`、`config.json`、`calibration_scores.csv`、`grouping.json`。确认 `config.json` 中 `test_evaluated=false`。

## 可以直接进行的非训练实验

主 checkpoint 已固定，因此可以立即运行尺度自适应 CQR validation-only 扫描：

```bash
cd /home/v2/.worktrees/ch4-hcq-cqr
/root/anaconda3/envs/torch23/bin/python \
  code/scripts/run_scale_adaptive_cqr_validation.py \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --device cuda
```

该脚本只使用 ratio=0.5 的 calibration/validation engines，不读取官方测试集。结果写入 `results_v4/scale_adaptive_cqr_validation/`。

## 禁止事项

- 不要把 `ratio=0.6/0.7` 的旧 41 epoch 结果与新的 patience=12 结果混成一张正式主表；
- 不要使用官方测试集选择聚合方式、epsilon、min group size 或 shrink tau；
- 不要把没有完成 `calibration_only.json` 的目录视为训练成功；
- 不要删除已有结果目录和其他 agent 的未提交修改；
- 训练完成后只汇报 validation 指标和 artifact 状态，等待主 agent 冻结最终配置。

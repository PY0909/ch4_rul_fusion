# 第四章 HCQ-CQR 实验交接

更新日期：2026-11-02

## 当前代码与结果

- A6-Point、A6 Quantile+CQR、HNN、EDL-NIG 的 FD002/FD004 seed 0 对比模型及产物在 `results_v4/chapter4_main/`。
- 第一阶段冻结模型校准矩阵在 `results_v4/calibration_stage1/`；包含 Block-Max、Late Endpoint、Weighted-Late 与 ratio 0.5/0.6/0.7 验证诊断。
- 上述第一阶段候选的测试矩阵和对比汇总在 `results_v4/calibration_stage1_test/`。官方测试发动机已经用于这些候选指标；不得再把它们当作未触碰的盲测数据，也不得据其结果继续挑选方案。
- 尺度自适应 CQR 的 12 个 validation-only 候选在 `results_v4/scale_adaptive_cqr_validation/`。主 checkpoint 使用 calibration ratio 0.5；脚本没有访问测试数据。
- 新校准器实现于 `rul_fusion/uncertainty.py`，扫描入口为 `scripts/run_scale_adaptive_cqr_validation.py`，方法矩阵为 `docs/scale_adaptive_cqr_stage2_experiment_matrix.md`。
- 完整续训命令和产物验收清单见 `docs/agent_training_resume_prompt.md`。

## 尺度自适应验证结果摘要

验证指标是每台 validation engine 的 latest (0.9 life fraction) cut。目标覆盖率为 0.8；结果为单 seed 诊断，不可视为稳定统计结论。

| 数据集 | 候选 | PICP | MPIW | Interval Score | 说明 |
|---|---|---:|---:|---:|---|
| FD002 | 原始 Block-Max | 1.000 | 63.48 | 63.48 | 覆盖过高、区间宽 |
| FD002 | 尺度自适应 Block-Max | 0.962 | 31.92 | 32.23 | 区间变窄，仍覆盖过高 |
| FD002 | 原始 Late | 0.731 | 12.93 | 18.68 | 欠覆盖 |
| FD002 | 尺度自适应 Late | 0.731 | 13.18 | 19.26 | 覆盖未改善 |
| FD002 | 原始 Weighted-Late | 0.923 | 16.43 | 19.08 | 覆盖偏高 |
| FD002 | 尺度自适应 Weighted-Late | 0.731 | 12.65 | 19.35 | 变窄但转为欠覆盖 |
| FD004 | 原始 Block-Max | 1.000 | 238.00 | 238.00 | 极宽、覆盖过高 |
| FD004 | 尺度自适应 Block-Max | 1.000 | 221.09 | 221.09 | 略变窄，覆盖不变 |
| FD004 | 原始 Late | 0.840 | 16.04 | 27.94 | 覆盖接近目标 |
| FD004 | 尺度自适应 Late | 0.840 | 16.04 | 27.94 | 与原始结果相同 |
| FD004 | 原始 Weighted-Late | 0.960 | 35.51 | 39.86 | 覆盖偏高 |
| FD004 | 尺度自适应 Weighted-Late | 0.960 | 32.76 | 33.33 | 区间变窄，覆盖仍偏高 |

当前没有证据说明尺度自适应 CQR 能稳定解决覆盖率问题。它在 Block-Max 上缩短了区间，在 Late 上没有可见收益；FD002 Weighted-Late 的区间变窄伴随欠覆盖。分层候选在部分设置中得到与全局候选相同的验证指标，不能仅因存在非空组阈值便声称分层有效。

## 训练续跑状态

目标是采用一致的早停协议训练 `FD002/FD004 × calibration_ratio {0.5, 0.6, 0.7}` 六个配置：`max_epochs=100`、`patience=12`、按 `val_q50_rmse` 选 checkpoint、`--defer_test`。每个 ratio 必须使用自己的 engine split，训练、validation、calibration engines 保持互斥。

当前不完整或不匹配的产物：

- `calibration_ratio_earlystop_v1/ratio_0.5/FD002` 曾中断在 epoch 5；没有 optimizer state，必须从头重跑。
- 该目录下 FD004 ratio 0.5 没有完成训练。
- `calibration_ratio_retrain/ratio_0.6` 和 `ratio_0.7` 的部分结果按 41 epoch 上限运行，不是 patience=12 的早停矩阵，只能作为探索产物。
- ratio 0.7 FD004 旧目录也没有完整 `calibration_only.json`，不能视作已完成。

因此建议训练 agent 按 `agent_training_resume_prompt.md` 的六配置串行重跑；完成后先核验 `best.pt`、`history.csv`、`calibration_only.json`、`calibrator.json`、`config.json`、`calibration_scores.csv`、`grouping.json` 均存在，且 `test_evaluated=false`。不能从旧的 `best.pt` 接着训练。

## 后续实验顺序

1. 完成六个统一早停训练，留存每轮 train loss、validation RMSE、best epoch 和划分 engine IDs。
2. 对每个匹配 checkpoint，仅在 validation 上重跑原始与尺度自适应 CQR。固定 eps、aggregation、coverage alpha 和分组策略；校准阈值只由该 split 的 calibration engines 拟合。
3. 用预先声明的规则选配置：优先要求 validation PICP 在 0.75–0.85，再在满足条件的候选中最小化 Interval Score；若无候选达标，报告 Pareto/完整结果，不选“最窄但欠覆盖”的区间。
4. 鉴于已有第一阶段测试结果被查看，后续不得反复根据 FD002/FD004 官方测试反馈调参。若论文需要确认性测试结论，需明确披露已有测试检查；更严格方案是另设独立外部数据/独立 split 或在论文中把后续结果标成探索性。
5. 当前只有 seed 0。冻结协议后，建议至少补充多 seed 结果并按发动机报告区间指标的不确定性；不要用重叠窗口数作为独立样本量。
6. Weighted-Late 是项目启发式聚合；分组阈值收缩也属启发式。对这些配置不要声称标准 split-conformal 的有限样本覆盖保证。Late Endpoint 只使用一个生命周期截点，也需明确其校准目标是对应的 endpoint estimand。

## 运行尺度自适应 validation 扫描

```bash
cd /home/v2/.worktrees/ch4-hcq-cqr
/root/anaconda3/envs/torch23/bin/python \
  code/scripts/run_scale_adaptive_cqr_validation.py \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --device cuda
```

输出每个数据集 12 个候选，并设置 `test_evaluated=false`。运行前确认 checkpoint 路径和 split 配置与结果矩阵一致。

## 验证

在交接时，`/root/anaconda3/envs/torch23/bin/python -m pytest -q code/tests` 通过：52 passed，1 个 PyTorch 卷积 padding warning。`git diff --check` 通过。

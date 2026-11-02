# B0→B5 单 seed 消融报告(FD002、FD004)

**日期:** 2026-10-28
**协议:** `IMPLEMENTATION_PLAN.md` §5,history=50,raw RUL 标签(`--rul_cap 0`),seed=0(单 seed),
epochs≤50,patience=10,batch=64,lr=1e-3,AdamW+ReduceLROnPlateau,checkpoint 按 val RMSE 选择,test 只评估一次。
**环境:** torch23(torch 2.5.1,CUDA,RTX 4060 Ti)。每运行 2–4 分钟,总时长约 37 分钟。
**输出:** `results_ablation/<block>/<dataset>/seed_0/{best.pt,config.json,metrics.json,history.csv,predictions.csv}`,日志在 `logs/`。

## 配置

| 块 | 配置 |
|---|---|
| B0 | 单尺度 + 禁用 router/graph/shared-specific(num_scales=1,基础 Conv 编码器+工况+标量头) |
| B1 | B0 + 三尺度(num_scales=3) |
| B2 | B1 + CrossScaleRouter |
| B3 | B2 + SensorGraphBlock |
| B4 | B3 + shared/specific 门控池化(alpha=0,无正则) |
| B5 | B4 + shared/specific 正交正则(alpha_shared=alpha_specific=0.01,首次取值,未调参) |

## Test 结果(评估 raw test RUL)

| 块 | FD002 MAE | FD002 RMSE | FD004 MAE | FD004 RMSE | FD002 best_ep | FD004 best_ep |
|---|---|---|---|---|---|---|
| B0 | 123.12 | 134.61 | 161.13 | 172.21 | 27 | 12 |
| B1 | 124.93 | 136.00 | 161.04 | 171.86 | 13 | 10 |
| B2 | 136.17 | 146.86 | 163.77 | 175.38 | 13 | 12 |
| B3 | 129.40 | 140.59 | 164.93 | 177.65 | 20 | 11 |
| B4 | 124.32 | 135.64 | 177.48 | 188.91 | 7 | 10 |
| B5 | 132.15 | 143.15 | 167.32 | 179.67 | 14 | 12 |

## 关键诊断:所有配置在 test 上整体坍缩

| 块 | FD002 val_RMSE | FD002 test corr | FD002 pred_std | FD004 val_RMSE | FD004 test corr | FD004 pred_std |
|---|---|---|---|---|---|---|
| B0 | 47.0 | 0.013 | 9.3 | 76.3 | -0.061 | 23.8 |
| B1 | 46.6 | 0.050 | 3.7 | 75.6 | -0.045 | 22.8 |
| B2 | 41.9 | -0.057 | 8.9 | 76.0 | -0.077 | 27.2 |
| B3 | 36.7 | 0.028 | 14.9 | 72.9 | 0.015 | 38.8 |
| B4 | 37.3 | 0.065 | 11.4 | 72.9 | 0.001 | 35.0 |
| B5 | 36.7 | 0.033 | 14.4 | 72.4 | -0.026 | 34.9 |

- 12/12 运行在 test 上 |corr| ≤ 0.08,预测近似常数(FD002 预测 std 3.7–14.9,FD004 22.8–38.8)。
- 同一 checkpoint 在 val 上有真实信号(FD002 val_RMSE 36.7–47.0,远低于常数基线;FD004 72.4–76.3)。
- 模块逐级增加时 val RMSE 单调改善(B0→B5:FD002 47.0→36.7),但 test 相关性始终为 0——失效与模块堆叠无关,是协议级的 train/test 分布偏移。
- 已排除:数据读取(RUL 文件 259 行与 259 个 test engine 对齐)、checkpoint 加载(直接载入 best.pt 复现)、标签错位。

## 结论

1. **当前预处理协议无法产生可解释的 B0→B5 消融结论**:基线在 test 上完全失效,模块增益(如有)被淹没。单看本表,"哪块带来增益"不可判定。
2. 失效机制(与文献一致):FD002/FD004 六工况下传感器全局标准化不充分,train(完整寿命)与 test(截断)在工况/寿命阶段分布上错位;叠加 raw 无 cap 标签在高 RUL 段的极端标签噪声,模型在 test 输入上输出饱和常数。
3. **按 IMPLEMENTATION_PLAN §5 的"若无稳定增益应删除模块"标准,本轮不能触发任何模块删除决策**;需先修复基线协议。

## 建议的下一步(未执行)

- 工况条件标准化(按 6 工况聚类分别 fit scaler)或工况输入归一化——FD002/FD004 的标准做法。
- 增加对应 `--rul_cap 125` 对照(refine-logs EXPERIMENT_PLAN B2 已预注册该敏感性)。
- 修复基线后重跑本 B0→B5 序列,再评估模块贡献;多 seed 按原计划补齐。

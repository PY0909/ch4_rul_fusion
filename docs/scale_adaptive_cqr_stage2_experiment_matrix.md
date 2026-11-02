# 尺度自适应 CQR 第二阶段实验矩阵

## 目标

评估按基础预测区间宽度归一化的 CQR 分数，是否改善 FD002、FD004 上的区间覆盖率与宽度折衷。当前阶段只用 calibration engines 拟合 conformal 阈值、validation engines 选择规则；不读取官方 test engines。

## 方法

令模型分位数为 `q10, q50, q90`，基础宽度 `w = q90 - q10`。单截点尺度自适应分数为：

```text
s = max(0, q10 - y, y - q90) / (w + eps)
```

先按发动机将四个固定生命周期截点聚合成一个校准分数，再对发动机级分数计算有限样本分位数。推理区间为：

```text
[q10 - qhat * (w + eps), q90 + qhat * (w + eps)]
```

默认 `eps=1e-6`。校准分数和区间扩张使用同一个 `eps` 与同一个预测宽度定义。层级模式沿用训练集拟合的 condition-stage grouper；组样本不足时回退到全局阈值。层级收缩是项目启发式，结果不应表述为标准分组 split-conformal 覆盖保证。

## 实验矩阵

FD002、FD004 分别执行以下 12 个候选。统一使用已有 seed 0 A6 Quantile checkpoint、`split_seed=0`、`calibration_ratio=0.5`、`fixed_cuts`、alpha=0.2；每台发动机四个生命周期截点的分数先聚合，再在独立 validation engines 的 0.9 生命周期截点评估。

| 编号 | 校准分数 | 发动机内聚合 | 阈值模式 | `min_group_size` | `shrink_tau` | 用途 |
|---|---|---|---|---:|---:|---|
| S0 | 原始绝对 CQR | Block-Max | Global | — | — | 当前协议基线 |
| S1 | 原始绝对 CQR | Late Endpoint | Global | — | — | 当前窄区间候选 |
| S2 | 原始绝对 CQR | Weighted-Late | Global | — | — | 当前加权聚合候选 |
| S3 | 归一化 CQR | Block-Max | Global | — | — | 尺度自适应主比较 |
| S4 | 归一化 CQR | Late Endpoint | Global | — | — | 尺度自适应 late 消融 |
| S5 | 归一化 CQR | Weighted-Late | Global | — | — | 尺度自适应加权消融 |
| S6 | 原始绝对 CQR | Block-Max | Hierarchical | 8 | 10 | 现有 HCQ 对照 |
| S7 | 归一化 CQR | Block-Max | Hierarchical | 8 | 10 | 尺度自适应分层候选 |
| S8 | 归一化 CQR | Late Endpoint | Hierarchical | 8 | 10 | 分层与 late 组合 |
| S9 | 归一化 CQR | Block-Max | Hierarchical | 10 | 20 | 小组收缩敏感性 |
| S10 | 归一化 CQR | Late Endpoint | Hierarchical | 10 | 20 | 小组收缩敏感性 |
| S11 | 归一化 CQR | Weighted-Late | Hierarchical | 10 | 20 | 分层加权消融 |

`Weighted-Late` 固定权重 `[0.05, 0.15, 0.30, 0.50]`，按生命周期顺序对应 0.25、0.50、0.75、0.90 截点。不得根据测试结果修改权重。

## 数据划分与训练

本矩阵复用现有 ratio=0.5 checkpoint，避免同时改变模型与校准方法。ratio=0.6、0.7 的划分敏感性不在本轮执行；若后续纳入，需针对各 ratio 重训并只用其对应 validation engines 选 checkpoint，再用完全独立 calibration engines 校准。检查 train、calibration、validation engine IDs 两两不重叠。

## 指标与选择

主要指标：validation latest-cut `PICP`、`coverage_gap`、`MPIW`、`IntervalScore`。目标覆盖率 0.8。先筛选 `0.75 <= PICP <= 0.85` 的候选，再按 Interval Score 升序选择；若无候选达到覆盖区间，报告全部候选与覆盖偏差，不把最窄区间直接称为最优。附报 Q50 RMSE，确保校准没有改变点预测。

必须记录：checkpoint 路径/epoch、数据划分 engine IDs、校准发动机数、每台发动机聚合分数、全局阈值、各组样本数和阈值、实际改变阈值的组数、EPS、测试集未评估标记。

## 最终测试门控

完成 validation 分析并冻结方法、聚合规则和分组参数之后，才允许对预先确定的最终候选评估官方 test 一次。测试结果只用于最终报告，不反馈调参。`ratio=0.5` 以外的候选需要相匹配的重训 checkpoint。

## 执行

```bash
/root/anaconda3/envs/torch23/bin/python code/scripts/run_scale_adaptive_cqr_validation.py \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --device cuda
```

结果写入 `code/results_v4/scale_adaptive_cqr_validation/`，每个数据集包含 `matrix.csv` 和 `summary.json`。本脚本仅加载 train、calibration、validation 数据集对象用于拟合训练分组器、校准阈值和验证指标，不会执行测试预测。

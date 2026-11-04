# 第四章模型对比与实验方案

## 1. 实验目标

第四章研究目标是评价完整的 **HCQ-CQR** 方法是否能够在多工况 RUL 预测中同时提高点预测精度、区间覆盖可靠性和传感器异常条件下的稳定性。

HCQ-CQR 的完整定义为：

```text
A6 多尺度时序/传感器图编码器
        + 非交叉 q0.1/q0.5/q0.9 分位数回归头
        + engine/unit 级 CQR 校准
        + 工况、退化阶段和传感器质量分层校准
        + 稀疏分组的全局阈值回退与收缩
```

主结果必须将 HCQ-CQR 作为一个完整方法，与近期完整概率 RUL 方法进行比较。使用相同 A6 编码器的实验只用于机制消融，不能替代完整模型之间的主对比。

## 2. 完整模型主对比

### 2.1 主结果模型

| 编号 | 方法 | 时间属性 | 不确定性机制 | 主要输出 | 实验定位 |
|---|---|---|---|---|---|
| M0 | A6-point | 第一章基线 | 无 | 单个 RUL | 点预测锚点 |
| M1 | CRULP-style | 2024 online / 2025 publication | Ensemble CQR + cross-conformal | 预测区间 | 与本文最接近的近期共形方法 |
| M2 | EDL-NIG RUL | 2024/2025 | Evidential Deep Regression | Student-t 预测分布、证据参数 | 近期证据概率模型 |
| M3 | Heteroscedastic Neural Network | 2025 | 条件均值和方差建模 | Gaussian 预测分布 | 参数化概率回归 |
| M4 | Deep Ensemble | 2025 benchmark | 多独立模型预测分布 | 均值、方差、经验分位数 | 强模型不确定性基线 |
| M5 | HCQ-CQR | 本文 | Quantile Regression + Hierarchical CQR | q10/q50/q90 和校准区间 | 本文完整方法 |

M0 用于确认第四章方法是否保持第一章的点预测能力。M1–M4 是完整概率模型或完整不确定性推理方法，原则上保留各自的预测头、损失函数和推理方式；不将它们强制替换为 A6 编码器。

### 2.2 近期方法的文献依据

- [A benchmark on uncertainty quantification for deep learning prognostics, Reliability Engineering & System Safety, 2025](https://ideas.repec.org/a/eee/reensy/v253y2025ics0951832024005854.html)：在 N-CMAPSS 上比较 BNN、MC Dropout、Deep Ensemble 和 Heteroscedastic Neural Network。
- [Remaining Useful Life Prediction with Uncertainty Quantification Using Evidential Deep Learning, 2024/2025](https://doi.org/10.2478/jaiscr-2025-0003)：在 C-MAPSS 上使用 EDL 进行 RUL 预测和不确定性量化。
- [CRULP: Reliable RUL Estimation Inspired by Conformal Prediction, IEEE TIM, 2024/2025](https://doi.org/10.1109/TIM.2024.3522678)：将 Ensemble CQR 与 cross-conformal prediction 用于 RUL 区间估计。
- [RobustUQ, Reliability Engineering & System Safety, 2025](https://ideas.repec.org/a/eee/reensy/v262y2025ics0951832025003783.html)：扩散模型缺失值修复结合共形预测，作为传感器异常鲁棒性扩展对照。

如果无法获得某篇论文的官方代码，应将对应结果标记为 `paper-method reimplementation`，并报告使用的架构、超参数和与原论文的差异，不直接复制原论文数值或声称复现官方 SOTA。

## 3. 方法实现要求

### 3.1 CRULP-style

训练多个独立 quantile 模型，得到每个模型的 q10、q50、q90，再进行 ensemble quantile aggregation 和 conformal calibration。若无法完整实现原论文的 cross-conformal 协议，则在结果中明确命名为 `CRULP-style Ensemble CQR`，不能简称为官方 CRULP。

### 3.2 EDL-NIG

模型输出 Normal-Inverse-Gamma 参数：

```text
gamma, nu, alpha, beta
```

使用 evidential regression loss，并由 NIG 后验得到预测均值、预测方差和预测区间。必须记录 evidence regularization 系数、数值稳定项和任何裁剪操作。EDL 结果同时报告点误差和区间校准，不能只报告预测精度。

### 3.3 Heteroscedastic Neural Network

模型输出条件均值 `mu` 和标准差 `sigma`，使用 Gaussian NLL：

```text
L = 0.5 * ((y - mu)^2 / sigma^2 + 2 log(sigma))
```

`sigma` 使用 Softplus 加最小正数保证数值稳定。RUL 非负约束、负下界处理和区间截断规则必须在实验前冻结，并在 manifest 中记录。

### 3.4 Deep Ensemble

每个 ensemble member 使用独立初始化、独立训练和独立模型参数。建议每个 run 使用 5 个 members，输出：

```text
预测均值 = members 的均值
预测方差 = members 间方差 + member 内部观测方差（若模型提供）
q10/q90 = member 预测经验分位数
```

Deep Ensemble 不复用 HCQ-CQR 的 calibration threshold；否则会把两种不确定性方法混合，无法解释结果。

### 3.5 HCQ-CQR

当前实现的 HCQ-CQR 固定为：

1. A6 编码器提取时序和传感器图表征；
2. 非交叉头输出 q10、q50、q90；
3. 使用 pinball loss 训练；
4. validation endpoint 只用于 q50 RMSE 早停和 checkpoint 选择；
5. calibration endpoint 独立计算 CQR score；
6. fixed-cuts 下每个 engine/unit 的多个端点先聚合为一个保守 score；
7. hierarchical 模式按工况、预测退化阶段和可观测传感器质量分组；
8. 稀疏或未见分组回退到全局阈值；
9. test 只在模型和校准器冻结后评估一次。

## 4. 固定 A6 的机制消融

该部分不用于替代主结果，而用于说明 HCQ-CQR 各模块的贡献。所有方法共享相同 A6 编码器、数据划分和训练预算。

| 编号 | 方法 | 改变的模块 | 目的 |
|---|---|---|---|
| A0 | A6-point | 无不确定性头 | 第一章点预测基线 |
| A1 | A6-Quantile | 单纯 q10/q50/q90 + pinball | 测量分位数头本身的效果 |
| A2 | A6-Quantile + Global CQR | 增加全局 CQR | 测量全局校准收益 |
| A3 | A6-Quantile + Hierarchical CQR | 增加工况/阶段/质量分层 | 测量 HCQ-CQR 分层机制收益 |
| A4 | A6 + MC Dropout | 30 次随机前向 | 与贝叶斯近似方法进行受控比较 |

主结论应根据以下链条判断：

```text
A1 相对 A0：分位数输出是否损害 q50 点预测？
A2 相对 A1：全局 CQR 是否改善 nominal coverage？
A3 相对 A2：分层校准是否改善重要子组 coverage 或 Interval Score？
A3 相对 M1–M4：完整 HCQ-CQR 是否具有竞争力？
```

## 5. 数据集与划分协议

### 5.1 数据集角色

| 数据集 | 角色 | 主要用途 |
|---|---|---|
| FD002 | 主实验 | 多工况、单故障模式，验证主要方法和分层校准 |
| FD004 | 主实验 | 多工况、双故障模式，检验结论是否依赖单一故障模式 |
| N-CMAPSS DS03 | 外部复核 | 更复杂的飞行工况和小样本 unit 级评估 |
| N-CMAPSS DS08a | 外部复核 | 不同运行配置下的迁移性检查 |

四个数据集分别训练、分别拟合 scaler、分别保存 checkpoint 和结果，不共享模型参数。

### 5.2 数据划分

```text
train engines/units
    -> 训练模型和拟合 scaler
validation engines/units
    -> 选择 q50 RMSE 最优 checkpoint
calibration engines/units
    -> 估计 CQR 或其他区间校准参数
test engines/units
    -> 最终只评估一次
```

划分要求：

- FD002/FD004 使用 engine 级划分；
- DS03/DS08a 使用 unit 级划分；
- train、validation、calibration、test 集合互不重叠；
- scaler 只使用 train engines/units 拟合；
- fixed-cuts 的重叠窗口不能视为独立校准样本；
- CQR 校准按 engine/unit block 聚合；
- FD 训练标签保留冻结的 cap=125 协议，同时保存 raw RUL 标签；
- N-CMAPSS 结果必须标记独立 unit 数量较少，不能作为强总体 coverage 保证。

### 5.3 训练和测试协议

- FD002/FD004：history=50、patch CNN、d_model=64、grouped scaler、41 epochs、Adam、lr=1e-3；
- DS03/DS08a：history=16、d_model=32、2 heads、2 routers、global scaler、weight decay=1e-3；
- 每种方法至少 3 个 model seeds；
- Deep Ensemble 的 member 数量、seed 列表和训练预算提前固定；
- 所有超参数在 test 解封前冻结；
- 每个 dataset/method/seed 只允许一次正式 test 评估；
- smoke、单 epoch 和调参结果只能作为诊断，不进入正式主表。

## 6. 实验矩阵

### E1：完整模型主对比

| 数据集 | 方法 | seed | 结果 |
|---|---|---:|---|
| FD002 | M0–M5 | 0,1,2 | 主结果 |
| FD004 | M0–M5 | 0,1,2 | 主结果复核 |
| DS03 | M0、M1、M2、M3、M5 | 0,1,2 | 外部小样本复核 |
| DS08a | M0、M1、M2、M3、M5 | 0,1,2 | 外部小样本复核 |

如果完整 Deep Ensemble 或 CRULP-style 的计算成本过高，应先完成 FD002/FD004，再将 DS03/DS08a 的对应方法降为附加实验，并在论文中说明。

### E2：A6 机制消融

| 数据集 | 方法 | seed | 目的 |
|---|---|---:|---|
| FD002 | A0–A4 | 0,1,2 | 完整模块归因 |
| FD004 | A0、A1、A2、A3 | 0,1,2 | 主要结论复核 |

### E3：传感器质量鲁棒性

主协议为 **clean calibration → corrupted test**，不能使用污染后的 test 标签重新校准。

| 扰动 | 建议强度 | 数据集 | 评估方法 |
|---|---|---|---|
| 随机缺失 | 5%、15% | FD002 | M0、M2、M3、M5 |
| 加性噪声 | 0.10、0.30 × train sensor std | FD002 | M0、M2、M3、M5 |
| 传感器偏置 | 0.25、0.50 × train sensor std | FD002 | M0、M2、M3、M5 |
| 传感器漂移 | 0.25、0.50 × train sensor std | FD002 | M0、M2、M3、M5 |
| 混合扰动 | 预注册组合 | FD002 | HCQ-CQR 与最佳近期基线 |

主要观察指标是 coverage gap、MPIW、Interval Score 和 q50 RMSE 的退化幅度。

### E4：协议敏感性

- raw RUL 与 cap-125 标签；
- global CQR 与 hierarchical CQR；
- endpoint-level 与 engine/unit block calibration；
- 不同 calibration ratio；
- 不同最小分组样本数；
- unit 等权与按窗口加权的差异。

E4 只在 FD002 上进行，避免扩大主实验规模。

### E5：分层可靠性分析

在不使用真实 test RUL 构造校准分组的前提下，按以下维度事后分析：

- 工况组；
- 预测退化阶段；
- 可观测传感器缺失率；
- 真实 RUL 阶段，仅作为事后解释变量。

每个子组至少报告 coverage、coverage gap、MPIW、Interval Score、样本量和 bootstrap 置信区间。

## 7. 指标体系

### 7.1 点预测指标

| 指标 | 适用方法 | 说明 |
|---|---|---|
| MAE | 所有方法 | q50、均值或预测均值与真实 RUL 的绝对误差 |
| RMSE | 所有方法 | 对大误差敏感 |
| NASA Score | FD002/FD004 | 仅用于 C-MAPSS，DS03/DS08a 不报告 |
| q50 paired error | 分位数方法 | 与 A6-point 做 engine/unit 级配对比较 |

### 7.2 概率分布指标

| 指标 | 适用方法 | 说明 |
|---|---|---|
| NLL | HNN、EDL、必要时 DE | 必须明确使用的概率分布 |
| CRPS | HNN、EDL、DE | 需要完整分布或 Monte Carlo 样本 |
| Pinball loss | Quantile、CQR、HCQ-CQR | 对 q0.1、q0.5、q0.9 分别计算并取均值 |

只有 q10/q50/q90 三个分位点时，不应直接宣称获得严格 CRPS。若要报告 CRPS，应使用完整 Gaussian、Student-t 或足够密集的 Monte Carlo 预测样本，并说明计算方式。

### 7.3 区间指标

固定目标覆盖率为 80%（alpha=0.2）：

| 指标 | 含义 |
|---|---|
| PICP | 真实 RUL 落入区间的比例 |
| MPIW | 平均区间宽度 |
| Coverage gap | `abs(PICP - 0.8)` |
| Interval Score | 同时惩罚区间过宽和漏覆盖 |
| 子组 PICP | 工况、阶段和质量分组覆盖率 |
| 子组 MPIW | 分组区间宽度 |

PICP、MPIW 和 Interval Score 必须从保存的 `predictions.csv` 重新计算，不能只读取训练过程中的缓存数值。

### 7.4 统计报告

- FD002/FD004 使用 engine 级 bootstrap；
- DS03/DS08a 使用 unit 级 bootstrap；
- 不将重叠窗口作为 bootstrap 单位；
- 报告 3 个 seeds 的均值、标准差和 bootstrap 置信区间；
- 同时报告每个数据集的独立结果，不将四个数据集简单合并成一个总平均；
- 对 q50 MAE/RMSE 使用 engine/unit 级 paired comparison；
- 对覆盖率差异报告 nominal 80% 与经验 coverage 的距离，不只报告区间越宽越“可靠”。

## 8. 结果判定规则

HCQ-CQR 只有在满足以下条件时，才能在论文中使用“优于 Global CQR”或“具有整体优势”的表述：

1. FD002 和 FD004 的 clean-test coverage 接近 80%；
2. 至少一个预注册的重要子组的 coverage gap 或 Interval Score 改善；
3. MPIW 没有无界膨胀；
4. q50 MAE/RMSE 没有出现不可接受退化；
5. 3 个 seeds 的改进方向基本一致；
6. 与近期完整方法相比，改进不是由更宽区间单独造成的。

如果只提高 PICP 但区间明显变宽，应表述为“覆盖率提高但效率下降”，不能称为全面优越。

## 9. 结果文件要求

每个正式 run 至少保存：

```text
best.pt
metrics.json
history.csv
predictions.csv
calibration_scores.csv
calibrator.json
grouping.json
config.json
manifest.json
```

`manifest.json` 必须记录：

- 数据集和方法名称；
- model seed 与 split seed；
- train/validation/calibration/test engine 或 unit IDs；
- scaler 拟合范围及其 train IDs；
- 模型版本和代码 commit；
- calibration ratio、alpha、分组参数；
- test 评估时间；
- corruption 类型和强度；
- test 是否已经解封评估。

## 10. 方法命名和论文表述边界

建议在论文中使用以下命名：

```text
A6-point
A6-Quantile
Global CQR
HCQ-CQR
HNN reimplementation
EDL-NIG reimplementation
Deep Ensemble reimplementation
CRULP-style Ensemble CQR
```

只有在完整复现论文模型、数据协议和评测方式后，才使用论文原方法名。若使用当前 A6 编码器替代原论文 backbone，应明确写成“method-style reimplementation”或“机制复现”，不能写成原论文模型的官方复现。

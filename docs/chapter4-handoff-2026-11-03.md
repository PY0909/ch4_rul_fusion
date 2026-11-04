# 第四章实验交接（2026-11-03）

## 0. 交接范围与运行约束

- 工作分支：`codex/ch4-hcq-cqr`
- 最近提交：
  - `b533980` — N-CMAPSS 完整比较代码管线、测试与文档
  - `6b240d6` — FD002/FD004 window-weighted CQR、集成消融与披露日志
- 最近验证：torch23 环境完整测试 **62 passed, 1 warning**。
- 本交接之后的实验约束：**仅单 seed（seed 0），不跑任何 Ensemble**，包括 Gaussian Deep Ensemble 和 quantile-head ensemble。
- 当前没有 N-CMAPSS v4 完整比较训练在跑。`results_v4/ncmapss_comparison/summary.csv` 只有表头；新 runner 已就绪但尚未执行。
- `best.pt` 等 checkpoint 被仓库 `.gitignore` 忽略；相应运行的配置、预测、指标、日志已提交。工作区只剩根目录两份未跟踪 PDF 附件，不属于代码/结果提交。

---

## 1. 排名规则

### 点预测

FD002、FD004 分别独立排名；每个指标单独排序，**越低越好**。主表采用官方 test 文件中的原始 RUL 标签。表中“rank”只表示单 seed、当前固定 split 下的描述性名次，不代表跨 seed 显著性。

### 区间预测

目标覆盖率 `1−α=0.80`。主排名使用：

1. **覆盖资格门槛**：精确 PICP 落在闭区间 `[0.75, 0.85]`；
2. **资格方法内排序**：Interval Score（IS）升序，越低越好；
3. 同时给出无门槛的 IS 全排序，避免只看覆盖资格而隐藏区间效率。

覆盖资格带是本项目的描述性操作门槛，不是统计显著性证明。所有结果是 seed 0 单次 test；没有多 seed 方差或 bootstrap 区间。

---

## 2. FD002 / FD004 点预测结果及排名

测试标签为官方原始标签；所有方法口径相同。训练 RUL cap 为 125。

### FD002（n=259）

| MAE 排名 | RMSE 排名 | NASA 排名 | 方法 | MAE ↓ | RMSE ↓ | NASA ↓ |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 1 | A6-Quantile q50 | **15.532** | **22.656** | **3086.1** |
| 2 | 3 | 2 | A6-Point | 15.683 | 22.831 | 3321.5 |
| 3 | 2 | 3 | HCQ-CQR hierarchical q50 | 15.815 | 22.789 | 3643.2 |
| 4 | 5 | 5 | HNN | 22.791 | 31.302 | 15875.6 |
| 5 | 4 | 4 | EDL-NIG | 22.845 | 30.524 | 14112.5 |

### FD004（n=248）

| MAE 排名 | RMSE 排名 | NASA 排名 | 方法 | MAE ↓ | RMSE ↓ | NASA ↓ |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | 1 | A6-Point | **16.977** | **24.468** | **3939.8** |
| 2 | 2 | 2 | A6-Quantile q50 | 16.979 | 24.475 | 3982.3 |
| 3 | 3 | 3 | HCQ-CQR hierarchical q50 | 17.196 | 24.705 | 3985.7 |
| 4 | 4 | 4 | HNN | 22.169 | 28.676 | 6203.6 |
| 5 | 5 | 5 | EDL-NIG | 27.220 | 34.095 | 26779.3 |

### NASA cost-optimal 解码（补充分析，不是另一个训练模型）

对已训练 A6-Quantile 的 q10/q50/q90，根据预先给定的 NASA 非对称代价、用预测分布数值求期望代价最小点；不使用真实 test 标签拟合偏移。它改变的是点决策规则，不改变模型或区间。

| 数据集 | 解码 | MAE ↓ | RMSE ↓ | NASA ↓ | 平均偏移 |
|---|---|---:|---:|---:|---:|
| FD002 | 标准 q50 | 15.532 | 22.656 | 3086.1 | — |
| FD002 | cost-optimal | 15.591 | **22.505** | **3071.4** | +1.58 |
| FD004 | 标准 q50 | 16.979 | 24.475 | 3982.3 | — |
| FD004 | cost-optimal | **16.857** | **24.012** | **3710.7** | +2.12 |

**写作约定**：主结果仍报告标准 q50；cost-optimal 单列为“NASA-cost-aware decision rule”，不得和未经调整的模型输出混成一个排名。

---

## 3. FD002 / FD004 区间预测结果及排名

格式：`PICP / MPIW / IS`。主排名只纳入 split-matched 的核心方法；ratio 0.6/0.7 的 stage-1 结果与 checkpoint 选择验证引擎重叠，列作探索结果，不作为确认性名次。

### FD002

| 合格排名 | 全体 IS 描述排名 | 方法 | PICP | MPIW ↓ | IS ↓ | 覆盖资格 |
|---:|---:|---|---:|---:|---:|---|
| **1** | **1** | **A6 + Window-Weighted CQR** | **0.7568** | **44.41** | **85.50** | ✓ |
| 2 | 5 | HNN | 0.7761 | 64.31 | 102.76 | ✓ |
| — | 2 | A6 + window_plain | 0.7413 | 42.62 | 85.96 | ✗，略低于门槛 |
| — | 3 | A6 + Global CQR / block-max | 0.9035 | 77.08 | 91.23 | ✗，过覆盖 |
| — | 4 | A6 + late endpoint | 0.6178 | 26.53 | 94.74 | ✗，欠覆盖 |
| — | 6 | HCQ-CQR hierarchical | 0.8958 | 84.89 | 95.20 | ✗，过覆盖 |
| — | 7 | EDL-NIG | 0.6602 | 53.72 | 123.10 | ✗，欠覆盖 |

Stage-1 split-matched weighted-late：`PICP=0.6602 / MPIW=30.03 / IS=92.07`，不合格。它的区间更窄，但不能凭更小宽度判胜。

### FD004

| 合格排名 | 全体 IS 描述排名 | 方法 | PICP | MPIW ↓ | IS ↓ | 覆盖资格 |
|---:|---:|---|---:|---:|---:|---|
| **1** | **1** | **A6 + Window-Weighted CQR** | **0.7863** | **56.77** | **95.17** | ✓ |
| 2 | 4 | HNN | 0.7742 | 64.54 | 105.09 | ✓ |
| — | 2 | A6 + window_plain | 0.8750 | 82.33 | 99.65 | ✗，过覆盖 |
| — | 3 | A6 + weighted-late，ratio=0.5 | 0.7460 | 45.96 | 96.66 | ✗，略低于门槛 |
| — | 5 | EDL-NIG | 0.5685 | 52.20 | 152.12 | ✗，欠覆盖 |
| — | 6 | A6 + Global CQR / block-max | 1.0000 | 248.44 | 248.44 | ✗，严重过覆盖/退化 |
| — | 7 | HCQ-CQR hierarchical | 1.0000 | 248.48 | 248.48 | ✗，严重过覆盖/退化 |

Stage-1 weighted-late ratio=0.7 的结果为 `0.7702 / 49.48 / 95.91`，若只按 test 数字排序会排在 Window-Weighted 之后、HNN 之前；但它与 checkpoint 选择验证引擎重叠，**仅探索性**，不纳入主排名。

### 区间结论

- Window-Weighted CQR 在两个数据集均通过项目覆盖带，且 **IS 是所有已持久化方法/变体中的最低值**。
- 合格的 split-matched 方法中，排名 FD002/FD004 均为第 1；HNN 均为第 2。
- “第 1”是当前 seed 0、固定协议下的描述性排序，不等价于多 seed 显著优越。
- Global CQR 与 HCQ 在 FD004 的宽度约 248、PICP=1.0，属于过宽退化；HCQ 分组没有实际改变阈值。

### Test 使用与方法选择边界（重要）

`results_v4/weighted_cqr/test/` 是官方 test 上的**一次性探索性比较**：测试标签用于计算并排序多个校准变体的 PICP/IS；因此这些数字可描述已有 test 结果，但不能再称为完全未触碰 test 的独立确认，也不能在看完这些排名后继续用同一个 test 集反复调参。`test_touches.log` 记录了触碰。后续任何参数/变体选择必须在 test 之外冻结；新 test 评估应作为最终一次性报告。

---

## 4. CRPS、集成消融与历史报告数字

### CRPS

| 方法 | FD002 CRPS | FD004 CRPS | 说明 |
|---|---:|---:|---|
| A6-Quantile q10/q50/q90 | **12.683** | **13.754** | `CRPS_3pt`：在三个分位点间线性插值、尾部平坦延拓的近似 |
| HNN | 16.334 | 15.880 | 高斯分布 CRPS |
| EDL-NIG | 17.156 | 20.677 | Student-t / NIG 输出的蒙特卡洛 CRPS |

**比较口径提醒**：A6 的 `CRPS_3pt` 不是完整拟合连续分布后的精确 CRPS；尾部平坦延拓是粗近似。可以作为初步分布评分，但论文中需标明近似定义，不应不加限定地宣称与解析高斯 CRPS 完全同精度。若要把 CRPS 作为核心结论，应扩展分位点/尾部建模并统一积分精度。

### 已跑的 quantile ensemble 消融（不代表后续继续跑）

FD002 seed0/1/2 的三成员分位数平均：RMSE `22.689`，Window-Weighted IS `86.10`、PICP `0.7606`；单模型 seed0 对应 RMSE `22.656`、IS `85.50`。三成员未改善该 seed0 结果。seed1–4 的单模型训练产物完整；但**未执行五成员聚合评估**。后续按用户限定不再安排任何 ensemble 实验。

### Gaussian Deep Ensemble 历史报告值（不可复核排名）

旧 PDF 汇总曾报告 FD002 RMSE `26.44`、区间 `IS/PICP=97.9/0.72`；FD004 RMSE `32.01`、区间 `132.2/0.835`。当前仓库没有这批 Gaussian Ensemble 的预测与 metrics 工件，故标作**报告中历史数字**，不纳入本交接的正式排名；后续也不补跑。

---

## 5. 标签 cap 与评估协议

- FD 训练数据 `rul_cap=125`；test 使用官方原始 RUL 标签，所有对比方法相同。
- 官方 test 中超过 cap 的引擎：FD002 `57/259=22.0%`，FD004 `67/248=27.0%`；原始标签最大约 194–195。
- `results_v4/chapter4_main_logs/panel_recount.json` 对全部已持久化点模型同时给出 raw 与 diagnostic capped 标签重算。capped 重算只是敏感性分析，**不是**任何方法原先使用的 test 口径。
- 不要把 cap=125 训练、raw test 的结果解释为模型已学会外推 `RUL>125`。写作需披露这一局限。

---

## 6. N-CMAPSS 当前结果与 v4 状态

### 已有 v3 单 seed 点预测（旧 split/config，只作背景，不与新注册矩阵直接混排）

| 数据集 | 方法 | MAE ↓ | RMSE ↓ | 说明 |
|---|---|---:|---:|---|
| DS03 | A6 | 9.673 | 14.529 | v3 已有结果 |
| DS03 | FC-STGNN | **8.198** | **11.062** | v3 已有结果 |
| DS08a | A6 | 14.220 | 18.067 | v3 已有结果 |
| DS08a | FC-STGNN | **11.499** | **14.967** | v3 已有结果 |

这些是旧 split 和旧训练配置的结果；第四章新矩阵使用 registered split（4 train / 3 val / 2 cal）并对齐 fixed-cuts test endpoint，须全部重跑后再比较。新 split 的校准 unit 只有 2 个，因此 N-CMAPSS coverage 是小样本诊断，不宣称强有限样本保证。

### v4 尚未运行

`results_v4/ncmapss_comparison/summary.csv` 目前只有表头。代码命令矩阵已就绪：`scripts/run_ncmapss_comparison.py`。N-CMAPSS 不报告 NASA score。

---

## 7. 后续实验矩阵（单 seed、排除 Ensemble）

统一约束：`seed=0`、不跑 Gaussian Deep Ensemble、不跑任何分位数 ensemble、不做多 seed；所有 test 指标只在 checkpoint/方案冻结后评估一次。

| 优先级 | 实验 | 数据/方法 | 配置与目的 | 判读规则 |
|---|---|---|---|---|
| P0 | **N-CMAPSS 完整对比矩阵** | DS03、DS08a；A6-Point、FC-STGNN、A6-Quantile+CQR、HNN、EDL-NIG | 用相同 registered split、fixed-cuts endpoints、seed0；A6 使用 d32/2-head/2-router。运行前核对 A6 weight_decay 与锁定配置一致（当前矩阵命令需显式确认 `1e-3`）。 | 先冻结配置，再统一评估；DS03 单工况与 DS08a 多工况分开报告，coverage 只作小样本诊断。不得纳入 Ensemble。 |
| P1 | **FD cap=0 口径对照** | FD002/FD004；A6-Point、A6-Quantile、HNN、EDL-NIG | 单 seed0，训练原始 RUL，不截断；不含 Ensemble。对比现 cap=125 arm，回答训练/评估目标错位对结果的影响。 | 所有方法同口径；不能只重训 A6 后与旧基线横比。raw-label arm 独立报告，不替换现有结果直到全方法完成。 |
| P2 | **Window-Weighted 机制消融** | FD002/FD004；seed0 的冻结 checkpoint | 比较 ratio features（q50、宽度、工况特征）、权重 clip 与 LR 正则；先实现相应 CLI/脚本开关。 | 所有候选预先登记；不按同一 test 的 PICP/IS 选超参。validation 终点若与 test estimand 不匹配，则只作诊断，候选并列报告或使用独立校准数据。 |
| P3 | **条件覆盖诊断** | FD002/FD004；已冻结 predictions | 按 stage/condition 分组报告 coverage 与样本数，CPU 后处理。 | 小组样本数明确列出；不据此在 test 上再调分组或阈值。 |
| P4 | **校准量敏感性（可选）** | FD002/FD004；单 seed0 | calibration_ratio 对比需重划 split、重训对应模型；不包含 CV+ 多折集成式方案。 | 关注小校准集波动并如实报告；不以 test 选 ratio。 |

### N-CMAPSS 启动命令（仅命令示例，不代表已执行）

```bash
cd /home/v2/.worktrees/ch4-hcq-cqr/code
PYTHONPATH=. /root/anaconda3/envs/torch23/bin/python \
  scripts/run_ncmapss_comparison.py \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --python /root/anaconda3/envs/torch23/bin/python --run
```

脚本按顺序运行矩阵；用户要求的排除 Ensemble 已体现在当前 matrix commands 中。DS A6 使用 d32/2-head/2-router、dropout=0.1、weight_decay=1e-3；FC-STGNN 复用第三章模型配置；所有方法显式 seed0、同 registered split。运行前仍应审阅打印命令，运行后汇总写入 `results_v4/ncmapss_comparison/summary.csv`，日志及 artifacts 以各方法输出目录为准。

---

## 8. 关键文件索引

- 正式 FD 主指标：`code/results_v4/chapter4_main/**/metrics.json`
- HCQ 指标：`code/results_v4/uncertainty/hierarchical/**/metrics.json`
- Window-Weighted / plain / block-max 矩阵：`code/results_v4/weighted_cqr/test/matrix.csv`
- Stage-1 匹配/探索结果：`code/results_v4/calibration_stage1_test/{FD002,FD004}/test_matrix.csv`
- Raw/capped 点指标：`code/results_v4/chapter4_main_logs/panel_recount.json`
- Cost-optimal 解码：`code/results_v4/chapter4_main_logs/cost_optimal.json`
- 量化集成 3-member 负结果：`code/results_v4/ensemble_cqr_3member/summary.json`
- test 触碰披露：`code/results_v4/chapter4_main_logs/test_touches.log`
- 代码/实验矩阵：`code/scripts/run_ncmapss_comparison.py`、`code/scripts/run_ncmapss_cqr.py`、`code/scripts/run_ncmapss_probabilistic.py`、`docs/chapter4-experiment-matrix.md`

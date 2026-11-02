# 阶段 2/3 报告：六配置重训 + CQR 重扫 + 预注册规则选型（2026-11-02）

**协议**：按 `docs/chapter4_hcq_cqr_handoff.md` 冻结配置重训 6 组
（FD002/FD004 × ratio 0.5/0.6/0.7，100ep/patience12/val_q50_rmse/defer_test/
batch100/seed=split_seed=0），对每组 matched checkpoint 跑 S0–S11 十二候选
validation-only 重扫（72 项评估，`test_evaluated=false` 全程）。
**选型规则（预注册）**：先筛 PICP∈[0.75,0.85]，再最小化 Interval Score；
无候选达标 → 报 Pareto，不硬选。

## 重训验收（6/6 PASS）

7 文件清单齐、`test_evaluated=false`、train/cal/val engine 三方互斥
（确定性重放验证 + n_calibration 对账）：

| 配置 | best_ep | val_q50_RMSE | engines tr/va/ca |
|---|---:|---:|---|
| FD002 r0.5 | 25 | 30.47 | 208/26/26 |
| FD002 r0.6 | 7 | 29.54 | 208/21/31 |
| FD002 r0.7 | 5 | 31.74 | 208/16/36 |
| FD004 r0.5 | 15 | 46.44 | 199/25/25 |
| FD004 r0.6 | 11 | 49.95 | 199/20/30 |
| FD004 r0.7 | 15 | 45.56 | 199/15/35 |

## 重扫结果按规则筛后

**FD004（有达标候选）**——Late 家族全部达标：

| ratio | 候选 | PICP | MPIW | IS |
|---|---|---:|---:|---:|
| 0.5 | late(raw/scale_adaptive, global/hier 同值) | 0.760 | 14.4–14.6 | 22.05–22.16 |
| **0.6** | **late × 3 变体（并列最优 IS）** | **0.850** | 24.9–26.8 | **41.78** |
| **0.7** | **late × 4 变体（并列最优 IS）** | **0.800** | **15.49** | **31.08** |

r0.7 的 late 家族 PICP 恰为 0.800、区间最窄（IS 31.08）；但注意其 validation
仅 15 台 engine（12/15=0.8，粒度粗）。r0.6 达标且 validation engine 更多（20）。

**FD002（无候选达标 → Pareto 分支）**：三个 ratio 下 late 家族的 PICP 为
0.692–0.731（欠覆盖）→ 0.905（r0.6,略过）→ 1.000（r0.7,全过），**直接跳过
0.75–0.85 带**（validation 仅 16–26 台，单台粒度 3.8–6.3%）。Pareto 前沿：
r0.7 late raw（PICP 1.000, IS 13.67）↔ r0.6 late raw（0.905, 17.57）↔
r0.5 late 家族（≤0.731, 21.7）。按规则**不硬选**，如实报告。

**机制结论（与 stage1 一衷）**：
- 尺度自适应 vs 原始分数：late 家族两者几乎同值（多数完全并列；FD002 r0.7
  raw 略优 13.67 vs 13.82）；仅 block_max 上明显收窄区间（65→35）但 PICP
  仍 1.000 不达标。**无稳定收益，不保留。**
- 分层 vs 全局：几乎所有设置数值相同（组未激活/回退全局）。**分层无证据。**
- block_max 全线 PICP=1.000 + 超宽区间：按规则全部出局。

## 按规则产生的建议（检查点 A 提交决策）

- **方法**：Late-Endpoint 聚合 + 全局阈值 + 原始分数（并列时取最简变体；
  尺度自适应与分层均无证据支持，作为已探索的负结果报告）。
- **ratio**：FD004 取 0.7（PICP 0.800、最窄）或 0.6（更稳的估计）；
  FD002 无达标 → Pareto 报告（论文如实呈现覆盖跳带现象与 engine 粒度原因）。
- 全部结论限于 seed 0 单 split（多 seed 已按用户指示取消）；validation
  engine 数少（15–26），PICP 粒度粗，已在解读中标注。

产物：`results_v4/scale_adaptive_retrain_scan/ratio_{r}/{DS}/matrix.csv|summary.json`
（含全部 72 行 + 阈值/组激活状态；`test_evaluated=false`）。

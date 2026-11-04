# 第四章后续实验矩阵（window-weighted CQR 定稿路线）

生成日期：2026-11-02。基线状态：`window_weighted CQR` 已在 FD002/FD004 test 上双入带且 IS 全 panel 最低
（85.5 / 95.2，seed 0 单次）；CRPS、cost-optimal、双口径重算已完成并入库。
本矩阵回答"审稿人会要求补什么"，按优先级排列；每项给出目的、成本、产出与运行方式。

约定：所有命令在 `/home/v2/.worktrees/ch4-hcq-cqr/code` 下执行，
`PY=/root/anaconda3/envs/torch23/bin/python`，`DATA=/home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset`。
test 触碰须追加到 `results_v4/chapter4_main_logs/test_touches.log`（脚本已自动写入）。

---

## M0 已完成（存档，不再重跑）

| # | 实验 | 结果 | 产物 |
|---|---|---|---|
| 0.1 | A6-Point / Quantile+Global CQR / HNN / EDL-NIG × FD002/FD004（seed 0） | 主 panel | `results_v4/chapter4_main/` |
| 0.2 | 窗级加权 conformal 4 变体 × 2 数据集（test 单次） | WIN-W 双入带，IS 最低 | `results_v4/weighted_cqr/test/` |
| 0.3 | HCQ 分层（负结果） | 分组未激活，退化为 global | `results_v4/uncertainty/hierarchical/` |
| 0.4 | 3 成员集成（FD002，负结果） | IS 86.1 vs 85.5，无增益 | `results_v4/ensemble_cqr_3member/` |
| 0.5 | CRPS（分位数积分） | 12.68/13.75，胜 HNN 22%/13% | `rul_fusion/metrics.py::quantile_metrics` |
| 0.6 | cost-optimal 点估计 | FD004 NASA 3982→3711（无标签） | `results_v4/chapter4_main_logs/cost_optimal.json` |
| 0.7 | raw/capped 双口径全 panel 重算 | A6 两口径均领先 | `results_v4/chapter4_main_logs/panel_recount.json` |
| 0.8 | PICP Wilson CI | WIN-W 唯一双数据集 CI 含 0.8 | 汇总中 |

---

## M1 稳健性闭环（写作前必做，eval-only，无重训）

目的：证明 WIN-W 不是 seed-0 偶然，且阈值的每个设计选择都有依据。

| # | 实验 | 变量 | 成本 | 产出/判据 |
|---|---|---|---|---|
| 1.1 | **多 seed 稳定性**（核心） | 用已完成 checkpoint seed1/2（FD002）重跑 `run_weighted_cqr.py --endpoint test`；FD004 如需成员另训 | ~0.5h GPU（复用现成 ckpt）/ FD004 若补训 +3h | 各 seed 的 IS/PICP；判据：FD002 三 seed IS 均低于 91.2（旧 CQR）且 PICP 全部入带 |
| 1.2 | 密度比特征消融 | 特征 ∈ {仅 q50, q50+width, q50+width+cond（全）, 仅 cond} | eval-only，每特征组 ~2min | 特征组 × IS/PICP 矩阵；判据：完整特征组不低于任何退化组 |
| 1.3 | 权重裁剪与 LR 正则敏感性 | clip ∈ {5,10,20,50}, C ∈ {0.1,1,10} | eval-only，12 组 × 2min | 阈值/IS 变化幅度；判据：默认组（clip=20, C=1）处于平台区 |
| 1.4 | α 敏感性 | α ∈ {0.1, 0.2, 0.3}（同一模型重校准，不重训） | eval-only ~10min | 三档 IS/PICP 曲线；支撑"方法不依赖 α=0.2 特调" |
| 1.5 | LOO 阈值稳定性 | 留一校准引擎重算分位数，得阈值分布 | eval-only ~5min | 阈值 95% 区间；判据：区间宽度 < 2× 默认阈值 |

```bash
# 1.1 示例（FD002 seed1 checkpoint）
PYTHONPATH=. $PY scripts/run_weighted_cqr.py --data_root $DATA --endpoint test \
  --checkpoint_root results_v4/chapter4_main/quantile_cqr/global_seed1  # 需先建软链或加 --checkpoint 参数
# 1.2–1.5 需给 run_weighted_cqr.py 增加 --features/--clip/--lr_c/--alpha/--loo 开关（半天工作量）
```

## M2 协议修正组（回应口径问题，重训量中等）

目的：把 train-cap=125 / test-raw 不一致从"缺陷披露"升级为"受控对比"。

| # | 实验 | 变量 | 成本 | 产出/判据 |
|---|---|---|---|---|
| 2.1 | **cap 一致性重训**（raw-label arm） | `--rul_cap 0`，A6-Quantile × FD002/FD004，seed 0 | 2×32min GPU | raw 口径下 MAE/RMSE/NASA；判据：raw NASA 显著低于 capped 模型的 3086/3982（预期 3700→2800 量级） |
| 2.2 | cap ∈ {100, 125, 150} 网格 | 同上，各 1 seed | 6×32min | cap-标签联合敏感性曲线；选定与文献可比口径 |
| 2.3 | 双口径主表定稿 | 重跑 `recount_panel_labels.py` 纳入 2.1/2.2 | CPU | 两张口径一致的主表 |

## M3 校准集规模组（治 n=26 之本，重训量大，可选）

| # | 实验 | 变量 | 成本 | 产出/判据 |
|---|---|---|---|---|
| 3.1 | calibration_ratio ∈ {0.3, 0.5, 0.7} × WIN-W | 需按新 ratio 重建 split 并重训 quantile（ratio 变了 split 就变） | 6×32min | n_cal ∈ {16,26,36} 下 WIN-W 的 IS/PICP；判据：IS 随 n_cal 单调改善，FD004 在 n_cal=36 时 PICP 仍入带 |
| 3.2 | CV+（10 折交叉 conformal，K=10 重训） | 每折留出校准引擎 | 20×32min ≈ 11h | 有效 n≈260 的阈值；判据：FD004 IS < 95.2 则替换主结果，否则作稳健性附录 |

## M4 方法强度组（把发现变成贡献，eval-only 优先）

| # | 实验 | 变量 | 成本 | 产出/判据 |
|---|---|---|---|---|
| 4.1 | **条件覆盖表**（test 端按 stage×condition） | 现 predictions.csv 分组统计 | CPU ~10min | 分组 PICP 表；判据：边缘入带且无单组 PICP<0.6 的塌陷（若有，成为 M5 动机） |
| 4.2 | 分组加权（组内密度比，替代 global 加权） | 按 stage 三组分别拟合 LR | eval-only ~15min | 对比 global 加权；判据：FD004 IS 改善 ≥5% 才纳入正文，否则入消融"无增益" |
| 4.3 | EDL-NIG 方差后校准（1 因子 isotonic/温度缩放） | 温度 τ ∈ 网格，校准集拟合 | eval-only ~20min | EDL PICP 0.66/0.57 → ≥0.75？；判据：能修则说明"基线欠覆盖是校准问题非架构问题"，堵审稿人 |
| 4.4 | 与 conformalized quantile regression (CQR+) / jackknife+ 对照 | 标准 CQR+ 公式实现于现有分数 | eval-only ~30min | IS 对比；判据：WIN-W 仍最低则主张"加权 + 窗级聚合均必要" |

## M5 外部与扩展（时间允许才做）

| # | 实验 | 变量 | 成本 | 产出/判据 |
|---|---|---|---|---|
| 5.1 | **N-CMAPSS 外部检查（DS08a 主 + DS03 次，与第三章数据集对齐）** | WIN-W + quantile 头迁移到 `build_ncmapss_datasets`（管道缺口，约半天工程量）；注册外部检查专用 split（dev 6/3，校准单元 ≥3，替换 registered 7/2 的单校准单元） | 1h 工程 + 4×~40min GPU（两数据集 × quantile 训练） | DS08a：多工况下 WIN-W 的 IS/PICP（方法机制的外部检验）；DS03：单工况，加权项预期退化为普通 conformal（方法边界样本）。点预测如实报（DS03 上 FC-STGNN 更强，不可回避）；判据：DS08a 区间指标与 FD002/004 结论方向一致即通过，DS03 仅报连续性 |
| 5.2 | FD001/FD003（单工况）泛化 | 全管线 × 2 数据集 | 4×20min | 完成四数据集主表；增强普适性主张 |
| 5.3 | 传感器缺失/腐蚀鲁棒性 | 复用 `rul_fusion/corruptions.py` + WIN-W | eval-only ~1h | 区间在高缺失下是否保覆盖；判据：PICP 退化幅度 < HNN/EDL |

---

## M5.1 N-CMAPSS external check (registered)

The external check uses `scripts/run_ncmapss_cqr.py` for DS03 and DS08a,
retaining the locked A6 `d_model=32`, 2-head, 2-router model and no RUL cap.
The data adapter supports an explicit registered split with three validation
units and two calibration units (minimum calibration guard), disjoint from
training. Both global CQR and window-weighted CQR are computed from reusable
helpers in `rul_fusion/uncertainty.py`; feature normalization is parameterized
rather than fixed to the FD cap of 125. `--defer_test` is the default-safe
workflow: official test data is not evaluated or logged until an explicit
non-deferred invocation. NASA score is not applicable to N-CMAPSS.



```
M1（半天，写作前必须） → M2.1（1h，口径定稿） → M4.1/4.3/4.4（1h，贡献加固）
→ 决策点：若 4.2 或 3.1 有增益 → 纳入；否则 M3/M4.2 写入消融负结果
→ M2.2/M3.2/M5（按审稿压力与机器时间排）
```


## N-CMAPSS full comparison matrix

`scripts/run_ncmapss_comparison.py --data_root "$DATA"` prints the DS03/DS08a
commands for existing A6-point and FC-STGNN, registered A6 CQR, and HNN,
EDL-NIG, and Gaussian ensemble. Add `--run` to execute after review. The
probabilistic runner defaults to deferred official-test access and requires
explicit `--eval_test`; the CQR command uses `--defer_test`. The summary is
written to `results_v4/ncmapss_comparison/summary.csv`. Probabilistic runs use
the registered 3-validation/2-calibration split, history 16, batch 32 and 300
epochs. NASA score is not reported for N-CMAPSS.


## 停止条件（预注册）

- 任一 M1 项使 FD002 WIN-W 的 IS 超过 91.2（旧 CQR）→ 主张降级为"带内且高效"仅 FD004 成立，须重写主结论。
- M2.1 raw 口径重训后点指标变差超过 5% → 保留 capped 口径为主表，raw 移附录。
- M4.2/3.1 无增益 → 按既有负结果惯例如实入档，不做参数搜索式补跑。

## 披露清单（随矩阵同步维护）

1. 单 seed 口径与 `test_touches.log` 的全部触碰记录；
2. 密度比使用无标签 test 协变量；
3. cap 口径双报告；
4. cost-optimal 为决策论变换（只用模型自身分布与已知代价函数）；
5. 集成 3 成员负结果与 8 成员计划取消的决策记录。

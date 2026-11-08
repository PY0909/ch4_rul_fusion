# N-CMAPSS 实验线归档说明

日期：2026-11-08。决定：N-CMAPSS（DS03/DS08a）实验线**关闭**，全部结果
保留为诊断性证据；第四章的外部确认改用 C-MAPSS FD001/FD003（另见后续
预注册）。本说明记录归档范围、结论与论文处理方式，防止选择性报告。

## 归档动机与诚信声明

NC 的最终臂结果（A6 在点预测与区间 IS 上均不敌 EDL-NIG/MSAUQ）**已被
观察之后**才做出放弃主线使用的决定。因此 NC 结果不得从论文中静默消失：
按计划 §8.4 的原定位（2 校准引擎的小样本外部诊断），以附录或一小节
如实报告，并引用本说明。

## 归档产物清单（全部已提交，hash 可溯）

- P0 完整对比：`results_v4/ncmapss_comparison/`（A6/FC-STGNN 点预测、
  CQR 三变体、HNN/EDL/B3-B6 概率基线、summary.csv）
- Arm B 扩截点：`results_v4/ncmapss_cqr_expanded/`（G9/G17，冻结
  checkpoint）
- dev 调参臂：`results_v4/ncmapss_a6_dev_tuning/`（lr/wd/dropout×2，
  q50-宽度失配证据）
- LOSO 链：`results_v4/ncmapss_frozen_diagnostics/`、`ncmapss_loso_search/`、
  `ncmapss_loso_budget_compare/`、`ncmapss_loso_baselines{,_v2,_corrected}/`、
  `ncmapss_loso_frozen/`、`ncmapss_loso_final/`、`ncmapss_loso_final_baselines/`、
  `ncmapss_loso_final_verification.csv`
- 协议文档：`docs/ncmapss-calibration-expansion-prereg-2026-11-05.md`、
  `ncmapss-a6-dev-tuning-2026-11-05.md`、`ncmapss-loso-preparation-
  amendment-2026-11-06.md`、`ncmapss-loso-budget-comparison-amendment-
  2026-11-07.md`、`ncmapss-loso-baseline-retraining-amendment-2026-11-07.md`、
  `ncmapss-loso-baseline-correction-amendment-2026-11-07.md`、
  `ncmapss-loso-selection-freeze-2026-11-07.md`
- test 触碰披露：`results_v4/chapter4_main_logs/test_touches.log`
  （NC 端点共 3 次触碰：P0、Arm B、LOSO 最终臂，全部留痕）

## 核心结论（论文附录可直接引用）

1. **P0**：NC 上概率基线反超主方法（DS03 最优为 B3 LSTM-prob 5.58，
   DS08a 为 HNN 14.73；A6-Quantile q50 为 19.81/16.38）。
2. **保形层塌缩**：2 校准引擎 × 4 截点 = 8 窗时，80% 有限样本分位数退
   化为最大分数，三个 CQR 变体阈值逐位相同（16.627/16.376）。
3. **Arm B（G9 扩截点）**：加权机制恢复分辨率；DS03 window-weighted
   IS 82.5→75.9，DS08a 反向（64.0→70.0）；引擎数（仍 2 台）是绝对瓶颈。
4. **dev 调参臂**：改善 q50 的配置同时崩塌分位数宽度（37.2→15.5），
   保形阈值被迫补偿，净区间更宽——仅按 q50 RMSE 选点不保区间质量。
5. **LOSO 最终臂**（7 台训练、冻结轮数、G9、第 3 次触碰描述性复核）：
   A6 + WW-CQR 区间 IS 三臂最优（DS03 74.1、DS08a 61.2），点预测无改善
   （20.0/17.2）；EDL-NIG（5.13/16.02）与 MSAUQ（9.36/13.25）仍全面
   领先。**NC 结论 = 方法边界证据**：小 fleet（4-7 训练引擎）+ 2 校准
   引擎结构性地不利于本方法的保形校准与多工况设计。

## 论文处理方式

- 附录/小节标题建议："N-CMAPSS 小样本外部诊断"。
- 必须披露：每子集仅 9 台开发引擎与 6 台测试引擎；test 端点 3 次触碰
  历史；LSTM-prob 在共同预算下未收敛（NLL 平台）。
- 不将 NC 数字与 FD 数字混排为同一协议下的排名。

## 不向 FD001/FD003 迁移的 NC 方法级调优

LOSO 预算机制（120/30 + 冻结轮数）、raw-IS 选型准则、G9 作为生产截点、
统一 300ep 基线预算——均为 NC 小 fleet 病理的补救，FD001/FD003（100
训练引擎/10 校准引擎）不存在这些病理。流程级改进（预注册、两段式
train/verify/evaluate、一次性 test 纪律、哈希核对）**保留并沿用**。

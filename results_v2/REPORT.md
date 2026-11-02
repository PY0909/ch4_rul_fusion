# 修复后重跑报告:B0–B5 × FD002/FD004(单 seed)

**日期:** 2026-10-28
**前置修复:** 测试窗口 bug(`_make_windows` final_only 按标签长度定终点,导致 test 输入为首周期重复窗口)已修复并加回归测试;修复前 12 组结果(`results_ablation/`)全部作废,仅存档。

## 本轮新增的协议组件

1. **Endpoint 验证(新增,checkpoint 选择依据):** 每个 validation engine 在其寿命的 uniform [0.5, 1.0] 随机分数处截断(独立 rng 流,split_seed 派生,可复现,截断点记录在 config),取截断末端窗口、标签为该点 **raw** 剩余 RUL(不受 rul_cap 影响)。
   - 实现说明:字面取"轨迹最后窗口"会使所有 val 标签恒为 0(val engine 来自 train 文件,终止于失效),是退化选择目标且与 test 估计目标(截断快照 + 剩余 RUL>0,FD002 均值 81)不符。截断式构造才是对 test 协议的忠实对齐。
   - checkpoint 选择、early stop、LR scheduler 均切换到 endpoint RMSE;全窗口 val RMSE 保留在 history.csv 作辅助指标。
2. **`--scaler grouped`:** 传感器按 setting1(round 1 位小数,恰为 6 工况组)分组 MinMax,仅 train engines 拟合;工况 3 列全局 MinMax(train 拟合);test 未见组显式回退全局统计并记录(`unseen_test_groups`,本轮 FD002/FD004 均为空)。cycle 列**未**加入工况输入(按用户要求留作单独消融)。
3. **标签协议:** train 端 cap=125(`--rul_cap 125`),**test 端始终评 raw RUL**(含 FCSTGNN 式 capped-test 口径的对照未在本轮展开)。

测试:13 passed(含新增 GroupedMinMaxScaler/endpoint 构造单测)。

## Phase 1–3:2×2 协议对照(B0 配置,seed=0)

test 评 raw RUL;val_ep = endpoint 验证 RMSE(checkpoint 选择准则)。

| 数据集 | scaler | cap=0 MAE/RMSE | cap=0 val_ep | cap=0 corr | cap=125 MAE/RMSE | cap=125 val_ep | cap=125 corr |
|---|---|---|---|---|---|---|---|
| FD002 | global | 35.30 / 45.25 | 27.59 | 0.563 | 34.91 / 44.23 | 26.38 | 0.583 |
| FD002 | grouped | 28.33 / 36.60 | 28.37 | 0.745 | **28.04 / 37.24** | **23.66** | **0.770** |
| FD004 | global | 41.37 / 50.72 | 48.30 | 0.575 | 36.48 / 44.45 | 38.36 | 0.590 |
| FD004 | grouped | 35.35 / 44.14 | 45.96 | 0.725 | **30.48 / 37.79** | **35.07** | **0.733** |

结论:
- **Sanity gate(raw+global,即仅修 bug 的对照):** FD002 corr 0.013→0.563、MAE 123→35.3;FD004 corr −0.061→0.575、MAE 161→41.4。坍缩消失,确认测试窗口 bug 是主因;归一化与 cap 是叠加改进而非根因(与"评估管线错误已确认、根因未确认"的判断一致,现已分别量化)。
- **cap=125(train 侧):** 4/4 组合 val_ep 与 test 双优,NASA 大幅下降(FD004 global 3.9M→99k)。
- **grouped:** 4/4 组合 test MAE/RMSE/corr 优;cap125 下 val_ep 双数据集优 → 按预定规则(val_ep 双数据集为准)选定 **cap125 + grouped** 为消融协议。

## Phase 4:B0→B5 消融(冻结协议:cap125 + grouped + endpoint-val 选择 + raw test,seed=0)

| 块 | FD002 MAE/RMSE/NASA | FD002 val_ep | FD004 MAE/RMSE/NASA | FD004 val_ep |
|---|---|---|---|---|
| B0 基线 | 27.48 / 36.33 / 49.5k | 24.73 | 29.84 / 37.51 / 21.8k | 35.14 |
| B1 +多尺度 | 27.62 / 37.08 / 55.7k | 22.97 | 21.95 / 29.38 / 6.9k | 26.81 |
| B2 +路由 | 18.99 / 27.74 / 17.1k | 17.32 | 21.17 / 28.05 / 5.3k | 26.85 |
| B3 +传感器图 | 17.53 / 26.81 / 15.2k | **15.02** | 20.23 / 27.86 / 5.7k | 27.06 |
| B4 +s/s 池化 | 17.73 / 27.06 / 19.0k | 13.79 | 19.67 / **27.78** / 5.6k | 27.94 |
| B5 +s/s 正则 | **17.51 / 26.54** / 11.4k | 14.34 | 20.63 / 27.92 / 6.0k | 27.16 |

相对 B0 的变化(val_ep / test RMSE):FD002:B1 −1.76/+0.75,B2 −7.41/−8.59,B3 −9.71/−9.52,B4 −10.94/−9.27,B5 −10.39/−9.79;FD004:B1 −8.33/−8.13,B2 −8.29/−9.46,B3 −8.08/−9.65,B4 −7.20/−9.73,B5 −7.98/−9.59。

### 单 seed 初步读法(不作删除决策,按 §5 需多 seed 确认)

- **CrossScaleRouter(B2)是唯一在两个数据集上 val 与 test 都大幅一致的增益模块**(FD002 val_ep −7.4、FD004 test −9.5)。
- **多尺度(B1)数据集依赖:** FD004 明显增益(val_ep −8.3、RMSE −8.1),FD002 上 val 小幅改善但 test 略差(±1 内,单 seed 噪声量级)。
- **传感器图(B3):** 两数据集 test 都有小幅稳定增益(FD002 −0.93 vs B2,FD004 −0.19 vs B2),FD002 val_ep 明显改善。
- **s/s 池化(B4)与正则(B5):** 两数据集方向不一致(FD002 val 更好、FD004 val 更差;test 在 B3/B4/B5 间差 ≤0.8 RMSE,处于噪声量级),单 seed 无法判定,留给多 seed。
- 最终形态候选:B5(FD002 最佳)与 B4(FD004 最佳)差异在噪声内;B2→B3 的图模块有弱稳定证据。

## 边界与遗留

- 全部单 seed(seed=0)、20% engine 留出验证(与文献全量训练不同协议,数字不与文献表混排);best_epoch 偏早(7–26),patience/scheduler 未针对 endpoint 准则重调,多 seed 前可考虑复核。
- FCSTGNN 头对头需另加 capped-test 口径列;cycle 工况列、N-CMAPSS、多 seed 为后续项。
- 产物:`results_v2/grid/`(2×2)、`results_v2/ablation/`(B0–B5)、`grid_summary.csv`、`ablation_summary.csv`;旧 `results_ablation/` 未改动,标记作废。

## 补充 2(同日晚):cycle 消融更正、B3×FCSTGNN 配方对照

**更正:** 上一节"cycle 工况列消融"无效——当时 CLI 的 `--condition_features` 未转发给 `build_cmapss_datasets`(argparse 收值、builder 仍用默认 settings),8 个"cycle"运行实为 B2–B5 settings 的同配置重跑(GPU 非确定性分叉)。cli.py 已修复并加配方开关,消融以 `cycle_ablation_v2/` 为准;旧 `cycle_ablation/` 保留仅作重跑噪声参考。前文"B5+cycle FD002 全场最佳(25.75)"等读法全部作废。

**重跑噪声(副产品):** 同配置同 seed 成对重跑的 |ΔRMSE| = 0.09–6.82(中位 ~0.9;最大 B2/FD002 27.74↔34.56,best_epoch 21↔8)。因此:**单 run 间 ≤2 RMSE 的差距都在噪声量级内**;此前 B3/B4/B5 之间 ≤0.8 的排序、以及任何"最佳臂"读法都不可作为模块取舍依据。B2(路由)相对 B0/B1 的 8+ RMSE 增益大于典型噪声,但 B2 恰是方差最大的臂,模块删除决策仍需多 seed。

### B3 × FCSTGNN 训练配方对照(控制实验:分离配方与架构差异)

配方 = Adam(lr 1e-3, 无 wd)、batch 100、41 epochs 无早停、无 scheduler、无梯度裁剪;数据协议与选择准则不变。仅用于归因,不构成方法主张。

| 臂 | FD002 MAE/RMSE/NASA | FD002 val_ep | FD004 MAE/RMSE/NASA | FD004 val_ep |
|---|---|---|---|---|
| FCSTGNN(28k 参数) | 17.74 / 26.45 / **6.5k** | **11.91** | 18.93 / 27.03 / 5.4k | **23.94** |
| B3 默认配方(AdamW+sched+clip) | 17.53 / 26.81 / 15.2k | 15.02 | 20.23 / 27.86 / 5.7k | 27.06 |
| B3 + FCSTGNN 配方 | **17.52 / 26.17** / 15.1k | 14.65 | **18.88 / 26.71** / **4.9k** | 26.17 |

读法(单 run,对照噪声带解读):配方统一后 B3 在两数据集 test MAE/RMSE 均追平或略超 FCSTGNN(Δ ≤ 0.5,在噪声内),FD004 NASA 也同量级;此前的可见差距大部分来自训练配方而非架构。FCSTGNN 仍在 val_ep 与 FD002 NASA 上领先。是否将 B0–B5 全表统一到该配方,建议先做多 seed 再定。

### 修正版 cycle 消融(`cycle_ablation_v2/`,cycle 列确认进入 4 维工况输入)

| 臂 | FD002 Δval_ep / ΔRMSE(vs settings) | FD004 Δval_ep / ΔRMSE |
|---|---|---|
| B2+cycle | +6.51 / +2.84(显著变差) | −0.26 / +0.73 |
| B3+cycle | +0.55 / −1.02 | +1.96 / −0.21 |
| B4+cycle | +2.15 / −1.82 | **+8.05 / +6.19(崩坏,best_ep=2)** |
| B5+cycle | +1.10 / −0.40 | −0.41 / −0.85 |

结论:cycle 列无一致增益且引入不稳定(B2 两数据集 test 均变差、B4/FD004 崩坏),维持**不进默认配置**;个别 test 改善(如 B4+cycle/FD002 25.24)伴随 val 变差,按 val 准则不采纳。与绝对年龄捷径的先验一致。所有 Δ 处于或接近噪声带,采纳/否决的最终判定同样留给多 seed。

## 补充 3:per-sensor patch CNN 编码归因(`encoder_probe/`,FCSTGNN 配方,seed=0)

实现:`--encoder patch_cnn`(SensorPatchStem,每传感器序列 2×Conv1d(k=2)+BN+ReLU,BN 统计在 batch×sensors 展平轴,仅吃传感器值,条件经线性投影后融合)与 `--input_norm bn`(仅在线性编码后加 BatchNorm 的 BN 探针)。测试 22 passed。

| 臂 | FD002 val_ep / RMSE / NASA | FD004 val_ep / RMSE / NASA |
|---|---|---|
| B0 linear | 23.30 / 39.43 / 93.5k | 34.18 / 35.56 / 17.3k |
| B0 + bn 探针 | 23.49 / 35.52 / 39.7k | 31.74 / 35.03 / 14.0k |
| **B0 + patch_cnn** | **16.33 / 31.73 / 22.5k** | **28.28 / 29.11 / 6.7k** |
| B3 linear(ref) | 14.65 / 26.17 / 15.1k | 26.17 / 26.71 / 4.9k |
| B3 + bn 探针 | 14.57 / 27.22 / 14.7k | 25.93 / 28.25 / 6.8k |
| B3 + patch_cnn | 13.85 / **25.62** / **6.0k** | **23.52** / 28.70 / 7.5k |
| FCSTGNN(参照) | 11.91 / 26.45 / 6.5k | 23.94 / 27.03 / 5.4k |

读法(单 seed,对照噪声带):
- **BN 假设不成立**:BN 探针在 B3 上两数据集 Δval_ep ≤0.3——FCSTGNN 的优势不来自 BatchNorm。
- **patch stem 的增益集中在 B0**:val_ep −7.0(FD002)/−5.9(FD004),远超噪声带——逐点线性投影确实缺失时间编码,stem 补上了。
- **叠加到 B3 后增益变小且方向不一**:FD002 val/test 双改善(NASA 15.1k→6.0k 追平 FCSTGNN),FD004 val 改善 −2.65(23.52,基本追平 FCSTGNN 23.94)但 test 变差 +1.99——多尺度卷积与 stem 功能重叠,且 Δ 在噪声带边缘。
- 编码器轴确认有效,进入多 seed 矩阵:`encoder ∈ {linear, patch_cnn}`(BN 臂按证据剔除);FD002 上 B3+patch 与 FCSTGNN 仍余 ~2 val_ep 差距,归因候选只剩其图池化/更小容量。

### patch_cnn 系列补全:B4/B5(同 FCSTGNN 配方,seed=0)

在配方匹配的 patch_cnn 系列内部比较(val_ep,选择准则):

| 臂 | FD002 val_ep / RMSE / NASA | FD004 val_ep / RMSE / NASA |
|---|---|---|
| B0+pcnn | 16.33 / 31.73 / 22.5k | 28.28 / 29.11 / 6.7k |
| B3+pcnn | 13.85 / 25.62 / 6.0k | 23.52 / 28.70 / 7.5k |
| B4+pcnn | **12.79** / 26.59 / 9.4k | 24.86 / 27.91 / 9.8k |
| B5+pcnn | 14.34 / 26.07 / 9.4k | 25.01 / 27.51 / 6.7k |
| FCSTGNN | 11.91 / 26.45 / 6.5k | 23.94 / 27.03 / 5.4k |

读法(单 seed):
- **B4+pcnn 把 FD002 的 val_ep 差距收窄到 ~0.9**(12.79 vs FCSTGNN 11.91,噪声带内);FD004 上 B3+pcnn 已与 FCSTGNN 持平(23.52 vs 23.94)。两数据集分别由不同臂最优(B4 vs B3),s/s 池化在 pcnn 下的增益方向依旧数据集不一致。
- **B5 正则(α=0.01)在 pcnn 下两数据集 val 均无增益**(14.34/25.01,均劣于 B4 或 B3),维持默认不启用。
- 至此配方匹配矩阵补全:`{B0,B3,B4,B5} × patch_cnn` 齐备;所有臂间 ≤1.5 val_ep 的排序仍是单 seed 读法,多 seed 前不作取舍。



### 同协议总表(seed=0,test 评 raw RUL)

| 模型/臂 | FD002 MAE/RMSE/NASA | FD002 val_ep | FD004 MAE/RMSE/NASA | FD004 val_ep |
|---|---|---|---|---|
| **FCSTGNN(参照)** | **17.74 / 26.45 / 6.5k** | **11.91** | **18.93 / 27.03 / 5.4k** | **23.94** |
| B3(最优 settings 臂) | 17.53 / 26.81 / 15.2k | 15.02 | 20.23 / 27.86 / 5.7k | 27.06 |
| B5 + cycle | 16.98 / **25.75** / 8.5k | 14.43 | 20.41 / 28.42 / 6.2k | 27.68 |

完整 11×2 矩阵见 `results_v2/fcstgnn_cycle_summary.csv`。

### 读法(单 seed,均为噪声敏感)

- **FCSTGNN 在同协议下是最强参照:** 两数据集 val_ep 与 test RMSE/MAE 均优于我们全部 B0–B5 臂(FD002 val_ep 11.9 vs 我们最好 13.8)。且其 NASA 显著更低、参数量约为我们 1/3,还不用工况输入。按 §6 要求其协议与参数量已记录在 `results_v2/fcstgnn*/config.json`。这不是泄漏差异造成的——两边同为 endpoint-val 选择、同 split。
- ~~cycle 工况列消融~~(该小节结论作废,见"补充 2"更正:flag 未接线,运行为同配置重跑)。
- 与 FCSTGNN 的差距提示:瓶颈在训练配方与传感器编码,已在"补充 2"用 B3×配方对照验证:配方统一后差距基本消失。

*(以下历史小节保留原文以供追溯;"cycle 消融"表见上文更正说明)*

<details>

**协议(同上)与 FCSTGNN 参照原始数字:**

| 模型/臂 | FD002 MAE/RMSE/NASA | FD002 val_ep | FD004 MAE/RMSE/NASA | FD004 val_ep |
|---|---|---|---|---|
| **FCSTGNN(参照)** | **17.74 / 26.45 / 6.5k** | **11.91** | **18.93 / 27.03 / 5.4k** | **23.94** |
| B3(最优 settings 臂) | 17.53 / 26.81 / 15.2k | 15.02 | 20.23 / 27.86 / 5.7k | 27.06 |

完整矩阵见 `results_v2/fcstgnn_cycle_summary.csv`(其中 cycle 臂无效)与 `results_v2/recipe_and_cycle2_summary.csv`(更正后)。

</details>


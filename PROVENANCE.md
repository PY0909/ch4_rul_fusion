# PROVENANCE

本仓库的代码基线来源、验证状态与执行约定。

## 1. 快照来源

- **来源机器**：交接人新机器（文档内路径 `/home/v2/code`、worktree
  `/home/v2/.worktrees/ch4-hcq-cqr`，交接文档所见；本机无法访问）
- **快照时间**：2026-11-02（本地 `~/研/code` 收到的副本；无 `.git`，纯文件快照）
- **内容**：rul_fusion 完整包（16 模块）、scripts、tests（12 文件）、docs
  （chapter4_hcq_cqr_handoff / agent_training_resume_prompt /
  scale_adaptive_cqr_stage2_experiment_matrix）、references、locked_configs.json
  与 results_v2/v3/v4 + results_ablation 产物（**含 15 个 .pt checkpoint**，
  总计约 14MB，一并入库作为交接时点证据）。
- **方向决定（2026-11-02，用户拍板）**：HCQ-CQR 替代 SC-Surv 成为第四章主线。
  SC-Surv 线归档于 `PY0909/ch4_mtkdg`（P1 结果保留为探索性证据）。

## 2. 验证状态（本地，2026-11-02）

- conda `kaf_profiti`（Python 3.11.15、torch 2.5.1、numpy 2.0.2、
  scikit-learn 1.6.1、h5py 3.14.0）：**49 passed / 3 skipped**。
  3 个 skip 均为 FC-STGNN 基线测试（上游代码构造时调用 `.cuda()`，需 GPU，
  与当前 6 配置重训无关；AutoDL 上可复跑）。
- 交接时对侧机器记录：52 passed（其 torch23 环境，含 CUDA 可用的 3 项）。
- `compileall` 通过；AST 级内部 import 闭环零缺失。

## 3. 数据与环境约定

- FD002/FD004：`--data_root` 指向含 `CMAPSSData/` 的目录（本地
  `~/研/kst_light_v2/dataset`，AutoDL `/root/autodl-tmp/kst_light_v2/dataset`）。
- N-CMAPSS：`--data_root/data_set` 下放 DS03/DS08a HDF5（本地暂无，外部
  检查阶段再取并校验）。
- 环境 of record（对侧）：`/root/anaconda3/envs/torch23`；本侧执行统一用
  AutoDL `kaf_profiti`（torch 2.5.1+cu124），GPU 必须（FC-STGNN 基线）。

## 4. 执行纪律（继承自交接文档，不许违反）

- 6 配置统一早停重训**从随机初始化**，`max_epochs=100, patience=12,
  val_q50_rmse, --defer_test`，串行；验收 7 文件 +
  `config.json: test_evaluated=false`。
- 校准阈值只用对应 split 的 calibration engines 拟合；配置选择按预注册
  规则（PICP∈[0.75,0.85] 先筛、再最小化 IntervalScore，无达标报 Pareto）。
- **官方 test 已被 stage1 候选看过**：不得再用于任何选择；最终评估一次，
  论文披露前视状态。
- 不把 ratio=0.6/0.7 的旧 41-epoch 结果与新的 patience=12 结果混表；
  Weighted-Late/分层收缩按启发式措辞，不宣称标准 split-conformal 保证。

## 5. 2026-11-03 对接人线路采纳（本次合并）

以包 `~/研/ch4-hcq-cqr-8382f94/code`（分支 `codex/ch4-hcq-cqr`，commits
`b533980`/`6b240d6`，62 tests）覆盖更新本仓库：新增 window-weighted CQR、
N-CMAPSS 比较管线、cost-optimal 解码等；交接文档入 `docs/`。本地验证
59 passed / 3 skipped（skip 为 FC-STGNN CUDA 测试）+ compileall 通过。

**证据线状态**：11-03 线为正式口径（Window-Weighted CQR 为当前正式第 1，
其选择过程见 `results_v4/chapter4_main_logs/test_touches.log` 披露）；
本仓库 11-02 产物（late 重训/冻结候选 test/Deep Ensemble）降级为历史证据
——Ensemble 按新约束全面禁跑，其数字为"历史报告值"。
**后续约束**：仅 seed 0；不跑任何 Ensemble；test 只在冻结后评一次。

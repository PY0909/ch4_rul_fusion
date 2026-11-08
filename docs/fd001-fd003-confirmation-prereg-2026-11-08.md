# FD001/FD003 外部确认臂预注册

日期：2026-11-08（任何 FD001/FD003 评估之前登记）。前序：
`docs/ncmapss-archive-note-2026-11-08.md`（N-CMAPSS 线归档）。

## 定位与一次性纪律

FD001/FD003 的官方 test 集**从未被本项目评估过**（处女 test）。本臂的
全部价值在于一次性：训练阶段不加载 test（`load_test=False`），冻结后
经哈希核对，以 `--eval_test` 显式授权做**单次**评估，全部触碰记入
`test_touches.log`。评估的行集在本文档预注册，评估后不得增删改选。

## 协议：与 FD002/FD004 注册协议完全一致（原始设计）

不引入任何 N-CMAPSS 调优（无 LOSO、无 raw-IS 选型、无 G9、无统一
300ep 基线预算）。仅继承 NC 链的**流程纪律**：预注册、两段式
train/verify/evaluate、哈希核对、一次性 test、披露。

- 数据：C-MAPSS FD001（100 训练/100 测试引擎，1 工况 1 故障）、
  FD003（100/100，1 工况 2 故障）；test = 每引擎终末窗，官方原始
  RUL 标签，评估一次。
- 划分：引擎级，split_seed=0，val_ratio=0.2，calibration_ratio=0.5
  → 每数据集约 80 训练 / 10 验证 / 10 校准引擎。
- 校准：4 个固定截点（0.25/0.5/0.75/0.9）× 10 引擎 = 40 校准窗；
  密度比特征 F5（q50/125、宽度/125、条件均值）、clip=20、C=1、
  α=0.2、保守原子——与生产配置逐项相同。
- 训练标签 cap=125；grouped scaler（单工况下自然退化为单一组，
  机械上等价全局 min–max；条件协变量近似常数——预期行为，非缺陷）。
- A6-Point：`rul_fusion.cli` 锁定参数（locked_configs.json，2026-10-29
  锁定）：history 50、cap 125、grouped、patch_cnn、A6、fixed_cuts、
  adam、wd 0、batch 100、epochs 41、patience 41、no_scheduler、
  no_grad_clip、d64/4h/4r/dropout 0.1；checkpoint 由验证端点 RMSE 选择。
- A6-Quantile：`rul_fusion.quantile_cli` FD 默认解析（与冻结
  chapter4_main checkpoint 的 config.json 逐项一致）：history 50、
  cap 125、grouped、d64/4h/4r、dropout 0.1、epochs 41、batch 100、
  lr 1e-3、wd 0、patience 41、calibration_mode global、α 0.2、
  condition_mode full；选择准则 val q50 RMSE（fixed cuts）。
- 六个外部基线：`rul_fusion.probabilistic_cli` 论文原生默认
  （PAPER_METHOD_DEFAULTS；B3 的 BayesOpt 值即 FD001 原生、B4 原文
  即 FD001/FD003——基线在主场运行，登记偏离较 FD002/FD004 更少）。

## 预注册报告行集（评估一次，全部并排报告）

1. A6-Point（点指标：MAE/RMSE/NASA）；
2. A6-Quantile q50（点指标）+ 四个 CQR 变体（block_max_plain、
   window_plain、window_weighted、block_max_weighted；各报 threshold/
   PICP/MPIW/IS/q50_RMSE）；
3. B1–B6 六基线（论文原生概率指标：MAE/RMSE/NLL/CRPS/PICP/MPIW/IS，
   NCAPSS 除外口径不适用）。

排名规则沿用全章注册规则：PICP∈[0.75,0.85] 资格带内按 Interval Score
升序；同时给无门槛全排序。单 seed=0、无 ensemble。

## 预期与判读（登记时判断，供事后对照）

- 单工况下条件协变量无信号；按 FD 消融（A3/M1）承重特征为 q50
  （阶段适配），预期 window-weighted 机制仍与 plain/global 分化。
- FD001/FD003 为饱和基准（近年 RMSE ≈ 11–13 / 12–15）；A6 点预测
  不预设优势，本臂论点与 FD002/FD004 相同为校准质量。
- 四子集合并表按"单工况（FD001/FD003）vs 多工况（FD002/FD004）"
  组织；FD002/FD004 数字沿用已冻结主表，不重跑；两协议间差异不作
  增益解读。

## 执行

- 训练车道：`scripts/run_fd_confirmation.py --stage train`（defer，
  不加载 test）→ `--stage verify`（哈希清单）→ 用户授权后
  `--stage evaluate --eval_test`（单次）。
- 产物根：`results_v4/fd_confirmation/`。
- 冒烟覆盖（--smoke，小 epoch/MC 上限）：仅用于合成数据测试，
  manifest 标 partial_smoke，不得作为结果。

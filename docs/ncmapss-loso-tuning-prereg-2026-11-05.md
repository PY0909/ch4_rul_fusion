# N-CMAPSS LOSO 选型与最终臂预注册

日期：2026-11-05（任何搜索运行之前登记）。前序：Arm A（P0 冻结）、
Arm B（docs/ncmapss-calibration-expansion-prereg-2026-11-05.md）、dev 调参
臂（docs/ncmapss-a6-dev-tuning-2026-11-05.md）。三臂结果全部保留并报；
本注册不改任何已冻结数字。

## 目标与不动项

针对三个已证瓶颈（4 台训练引擎过拟合、3 台验证引擎选型信号弱且只看
q50、2 台校准引擎阈值退化），在不改动 A6-Quantile + Window-Weighted CQR
结构的前提下：以 LOSO 划分扩训练/选型信号，以激活函数与超参小网格搜
索，以含区间质量的选择准则修正 q50-宽度失配。

**不动**：A6 架构（d32/2 头/2 路由/patch_cnn/moment+level+single/趋势
特征）、q10/q50/q90 非交叉头、WW-CQR 规则（特征 F5、clip=20、C=1、
α=0.2、保守原子）、seed 0、无 ensemble、官方 test 评估仍为每臂一次性。

## 可动轴（网格预注册）

- 激活函数：legacy（=现状混合 ReLU/GELU，逐位基线）/ gelu / silu，
  统一替换模型与分位数头的全部非线性；
- 学习率：1e-3 / 3e-4；dropout：0.1 / 0.2；weight decay 固定 1e-3；
- epoch 预算：不进网格，由 LOSO 曲线后验选取（见规则）；
- history、batch(32)、优化器(Adam)、损失(pinball+reg) 与生产一致。

**S1 筛选网格**（dropout 0.1）：{legacy,gelu,silu} × {1e-3,3e-4} = 6 配置。
**S2 精化**：S1 前二配置 × dropout 0.2（以及必要时的 lr 交叉复核），
仅按 LOSO 聚合结果决定，不看 test。

## LOSO-CV 协议（选型层，不触 test）

9 台 dev 引擎，每折留 1 台：训练 = 其余 8 台（`unit_override`:
train=8, val=[留出], calibration=[]）。每折每配置训练一次（上限 150
epoch，batch 32，seed 0），逐 epoch 在留出引擎的 4 个 fixed-cuts 端点上
记录：三分位数 pinball 均值（主准则）、q50 RMSE、原始（无保形）80%
Interval Score。聚合 = 9 折逐 epoch 均值。

**预算规则**：每配置的 epoch 预算 = 主准则折均值曲线的 5-epoch 滑动平均
最小点。**配置排序规则**（在各配置自身预算处比较）：主准则均值升序，
平局依次比 q50 RMSE、原始 IS。

## 最终臂（LOSO-tuned）

超参、激活、epoch 预算冻结后：每数据集训练一个最终模型，
`unit_override`: train = 9 − 2 注册校准引擎 = 7 台（原 4 训练 + 原 3 验证
——验证引擎历史上仅用于选点，转为训练数据并在披露中说明），val=[]，
calibration = 注册 2 台校准引擎（保持与 P0/Arm B 相同引擎集合），固定
epoch 预算、无早停。校准 = G9 扩展截点（9 截点/引擎，Arm B 已证恢复加
权机制分辨率）；WW-CQR 三变体照常计算。

**评估**：官方 6 台 test 引擎 × 4 截点，一次性，test_touches.log 披露
（该端点历史上已被 Arm A/Arm B 触碰，本臂为第 3 次；论文三臂并排，
LOSO-tuned 的 test 数字不构成独立确认）。

## 判读边界

LOSO 聚合（9 折）是 dev 内泛化的诚实估计；最终 test 仅描述性。若
LOSO-tuned 不优于 P0，如实报告并保留 P0 为正式结果。预期登记（供事后
对照）：8 台训练下分位数头过拟合显著缓解；含区间质量的选型准则下，
低 lr/高 dropout 不再以区间崩塌换取 q50；G9 校准使加权与 global 阈值
继续分化。

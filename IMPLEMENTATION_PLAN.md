# 多尺度路由传感器图 RUL 实现方案

## 1. 研究对象

四个数据集分别训练、分别保存参数和 scaler：

- C-MAPSS FD002、FD004：预测单位为 cycles。
- N-CMAPSS DS03、DS08a：预测单位为 flights。

不共享跨数据集权重。四个数据集只共享代码结构和训练接口。

## 2. 模型结构

输入统一为：

```text
x: [batch, history, sensors]
w: [batch, history, operating_conditions]
```

模型依次执行：

1. 将每个传感器的观测与同一时刻工况拼接，线性投影到 `d_model`。
2. 沿时间轴构造原始、1/2、1/4 三个尺度；每个尺度用 Conv1d 和时间平均池化得到传感器 token。
3. 对 `[batch, sensors, scales, d_model]` 使用 sender/receiver router。该模块借鉴 PhaseFormer 的信息流，但 token 轴是时间尺度，不是物理 phase。
4. 将尺度池化后的传感器 token送入一层自适应传感器图。邻接矩阵由可学习源/目标节点嵌入生成，并添加自环。
5. 用门控投影分解 shared/specific 表征，并保留图池化、末端状态和工况摘要。
6. 通过两层 MLP 和 Softplus 输出一个非负 RUL 标量。

`alpha_shared` 和 `alpha_specific` 默认均为 0。启用后只作为消融，不能在查看测试集后调节。

## 3. 数据处理

### FD002/FD004

- 读取 `train_FD00x.txt`、`test_FD00x.txt`、`RUL_FD00x.txt`。
- 使用 14 个非恒定传感器和 3 个 operating settings。
- 训练标签为 `max_cycle(engine)-cycle`，`--rul_cap 0` 表示不截顶。
- 先按 engine 划分 train/validation，再生成窗口；默认 validation 比例为 20%。
- 默认历史长度为 50，短序列左侧复制首行填充。
- scaler 只在 train engine 上拟合。
- 每个 test engine 只保留最后一个窗口。

### DS03/DS08a

- HDF5 采用 chunk 读取，避免同时加载所有数组。
- 按 `(unit, cycle)` 聚合高频行；传感器和工况使用周期均值。
- 周期内 `Y` 必须恒定，否则直接报错。
- `T_*` 被识别为退化/健康参数，当前排除，避免标签泄漏；代码记录 `uses_T=false`。
- dev units 按完整 unit 划分 train/validation，默认 80/20；官方 test units 锁定。
- 默认只使用 `X_s`，`--include_virtual_sensors` 用于物理/虚拟传感器消融。
- 每个 test unit 只保留最后一个 flight-cycle 历史窗口。

## 4. 训练和输出

验证集 RMSE 用于 checkpoint 选择；训练期间不读取 test：

```text
epoch:1, train_loss:..., val_loss:..., test_loss:deferred
```

训练结束后只评估一次 test，并输出：

```text
MAE: ... RMSE: ... NASA_Score: ...
```

每个运行目录生成：

```text
best.pt
config.json
metrics.json
history.csv
predictions.csv
```

## 5. 建议实验顺序

固定数据划分、seed 集合、训练预算和评价脚本，依次运行：

```text
B0: TCN/当前 Conv 编码器 + 工况 + 标量头
B1: B0 + 多尺度
B2: B1 + CrossScaleRouter
B3: B2 + SensorGraphBlock
B4: B3 + shared/specific pooling
B5: B4 + shared/specific 正则
```

每一步至少使用 3 个 model seed。FD 数据按 engine bootstrap，N-CMAPSS 按 unit bootstrap。若一个模块在多个 seed 上没有稳定增益，应从最终模型中删除。

## 6. 近三年对照

正式表格只放同一 runner 重跑或协议完全核对过的结果：

- GRU/LSTM、TCN：基础序列基线。
- PatchTST、TimesNet、iTransformer：通用时序模型。
- 多工况 MoE/encoder mixture：工况建模对照。
- THGNN/自适应图卷积：图结构对照。
- Mamba/SSM：近年状态空间对照。

论文原始表格只能作为背景，不能和本实现直接混排。每个对照必须记录 split、RUL cap、窗口、特征、scaler、checkpoint 指标、seed 和参数量。

## 7. 运行命令

```bash
PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m rul_fusion.cli \
  --dataset FD002 \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --history 50 --rul_cap 0 --seed 0 --output_dir /home/v2/code/results

PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m rul_fusion.cli \
  --dataset DS03 \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --history 16 --seed 0 --output_dir /home/v2/code/results
```

## 8. 当前验证边界

当前已完成接口测试、真实数据读取审计、合成数据训练 smoke 和真实 FD002 单 epoch CPU smoke。尚未运行四个数据集的正式多 seed 训练，因此当前代码不包含、也不能宣称任何 SOTA 结果。

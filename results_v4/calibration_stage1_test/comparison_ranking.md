# FD002 / FD004：第四章对比模型与第一阶段校准方案测试集汇总

统一列出已有对比模型和第一阶段全部校准候选。MAE 排名按升序；区间方法另按 Interval Score 升序排名。PICP 目标为 0.8，NASA Score 越低越好。所有结果均为 seed 0 单次结果。

ratio=0.6/0.7 的校准方案与现有 checkpoint 选 epoch 所用验证发动机有重叠，标为探索性，其排名仅描述测试集数值，不用于选模。层级扫描 6 组参数逐一列出；本批测试中实际改变阈值的组数为 0。A6 Quantile + Global CQR 与 Block-Max ratio=0.5 是同一结果，合并为一行。

## FD002

| MAE rank | Interval Score rank | Method | Type | MAE | RMSE | NASA Score | PICP | MPIW | Interval Score | NLL | CRPS | Split/group note |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | A6-Quantile + Global CQR (Block-Max, r=0.5) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.903 | 77.08 | 91.23 | — | — | ratio=0.5 matched; — |
| 1 | 2 | A6-Quantile + Weighted-Late (r=0.7) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.668 | 30.65 | 91.65 | — | — | exploratory: validation split overlap; — |
| 1 | 3 | A6-Quantile + Weighted-Late (r=0.6) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.668 | 30.32 | 91.87 | — | — | exploratory: validation split overlap; — |
| 1 | 4 | A6-Quantile + Weighted-Late (r=0.5) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.660 | 30.03 | 92.07 | — | — | ratio=0.5 matched; — |
| 1 | 5 | A6-Quantile + Late Endpoint (r=0.6) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.656 | 29.13 | 92.70 | — | — | exploratory: validation split overlap; — |
| 1 | 6 | A6-Quantile + Late Endpoint (r=0.7) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.656 | 28.77 | 92.96 | — | — | exploratory: validation split overlap; — |
| 1 | 7 | A6-Quantile + Late Endpoint (r=0.5) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; — |
| 1 | 7 | HCQ-CQR Late (min=10, tau=10) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; 0 active groups |
| 1 | 7 | HCQ-CQR Late (min=10, tau=20) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; 0 active groups |
| 1 | 7 | HCQ-CQR Late (min=15, tau=10) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; 0 active groups |
| 1 | 7 | HCQ-CQR Late (min=15, tau=20) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; 0 active groups |
| 1 | 7 | HCQ-CQR Late (min=8, tau=10) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; 0 active groups |
| 1 | 7 | HCQ-CQR Late (min=8, tau=20) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.618 | 26.53 | 94.74 | — | — | ratio=0.5 matched; 0 active groups |
| 1 | 10 | A6-Quantile + Block-Max (r=0.7) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.946 | 101.57 | 106.66 | — | — | exploratory: validation split overlap; — |
| 1 | 11 | A6-Quantile + Block-Max (r=0.6) | Calibration/interval method | 15.53 | 22.66 | 3086.05 | 0.946 | 104.81 | 109.02 | — | — | exploratory: validation split overlap; — |
| 2 | — | A6-Point | Point model | 15.68 | 22.83 | 3321.50 | — | — | — | — | — | ratio=0.5 matched; — |
| 3 | 8 | HCQ-CQR (existing hierarchical setting) | Probabilistic baseline | 15.81 | 22.79 | 3643.23 | 0.896 | 84.89 | 95.20 | — | — | ratio=0.5 matched; — |
| 4 | 9 | HNN | Probabilistic baseline | 22.79 | 31.30 | 15875.64 | 0.776 | 64.31 | 102.76 | 4.688 | 16.33 | ratio=0.5 matched; — |
| 5 | 12 | EDL-NIG | Probabilistic baseline | 22.85 | 30.52 | 14112.47 | 0.660 | 53.72 | 123.10 | 4.894 | 17.16 | ratio=0.5 matched; — |

## FD004

| MAE rank | Interval Score rank | Method | Type | MAE | RMSE | NASA Score | PICP | MPIW | Interval Score | NLL | CRPS | Split/group note |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | — | A6-Point | Point model | 16.98 | 24.47 | 3939.78 | — | — | — | — | — | ratio=0.5 matched; — |
| 2 | 1 | A6-Quantile + Weighted-Late (r=0.7) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.770 | 49.48 | 95.91 | — | — | exploratory: validation split overlap; — |
| 2 | 2 | A6-Quantile + Weighted-Late (r=0.5) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.746 | 45.96 | 96.66 | — | — | ratio=0.5 matched; — |
| 2 | 3 | A6-Quantile + Weighted-Late (r=0.6) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.742 | 45.61 | 96.75 | — | — | exploratory: validation split overlap; — |
| 2 | 5 | A6-Quantile + Late Endpoint (r=0.5) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; — |
| 2 | 5 | A6-Quantile + Late Endpoint (r=0.6) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | exploratory: validation split overlap; — |
| 2 | 5 | A6-Quantile + Late Endpoint (r=0.7) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | exploratory: validation split overlap; — |
| 2 | 5 | HCQ-CQR Late (min=10, tau=10) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; 0 active groups |
| 2 | 5 | HCQ-CQR Late (min=10, tau=20) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; 0 active groups |
| 2 | 5 | HCQ-CQR Late (min=15, tau=10) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; 0 active groups |
| 2 | 5 | HCQ-CQR Late (min=15, tau=20) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; 0 active groups |
| 2 | 5 | HCQ-CQR Late (min=8, tau=10) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; 0 active groups |
| 2 | 5 | HCQ-CQR Late (min=8, tau=20) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 0.617 | 26.48 | 107.03 | — | — | ratio=0.5 matched; 0 active groups |
| 2 | 7 | A6-Quantile + Block-Max (r=0.6) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 1.000 | 200.82 | 200.82 | — | — | exploratory: validation split overlap; — |
| 2 | 8 | A6-Quantile + Global CQR (Block-Max, r=0.5) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 1.000 | 248.44 | 248.44 | — | — | ratio=0.5 matched; — |
| 2 | 10 | A6-Quantile + Block-Max (r=0.7) | Calibration/interval method | 16.98 | 24.47 | 3982.44 | 1.000 | 267.42 | 267.42 | — | — | exploratory: validation split overlap; — |
| 3 | 9 | HCQ-CQR (existing hierarchical setting) | Probabilistic baseline | 17.20 | 24.70 | 3985.71 | 1.000 | 248.48 | 248.48 | — | — | ratio=0.5 matched; — |
| 4 | 4 | HNN | Probabilistic baseline | 22.17 | 28.68 | 6203.61 | 0.774 | 64.54 | 105.09 | 4.780 | 15.88 | ratio=0.5 matched; — |
| 5 | 6 | EDL-NIG | Probabilistic baseline | 27.22 | 34.10 | 26779.32 | 0.569 | 52.20 | 152.12 | 5.302 | 20.68 | ratio=0.5 matched; — |

Point-only models have no interval rank. NLL/CRPS appear only for methods with persisted density scores; quantile/CQR methods do not define them. MAE ranks cover all rows; Interval Score ranks cover rows with interval metrics. Ties share rank. Do not select or tune methods using test ranks.

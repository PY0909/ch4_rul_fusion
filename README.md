# Multi-Scale Routed Sensor-Graph RUL

This directory contains an independent point-prediction implementation for
C-MAPSS FD002/FD004 and N-CMAPSS DS03/DS08a. It borrows mechanisms, rather
than source code, from the three reference repositories:

- DisMS-TS: temporal multi-scale representations and shared/specific pooling.
- PhaseFormer_TSL: sender/receiver router attention, renamed here to
  `CrossScaleRouter` because its tokens are temporal scales, not physical phase.
- FCSTGNN: lightweight adaptive sensor-node message passing.

The fusion model does not modify or import the three reference repositories;
the models and scalers are trained independently for each dataset. The single
exception is the baseline adapter below, which loads the unmodified FC-STGNN
model file so the baseline comparison runs the exact published architecture.

## Model

The legacy core model is:

```text
sensor sequence + operating conditions
  -> condition-aware multi-scale Conv1d encoder
  -> cross-scale router attention
  -> one adaptive sensor graph block
  -> shared/specific pooling + condition summary + last-state summary
  -> Softplus scalar RUL head
```

The default model uses three scales, `d_model=64`, four attention heads, four
routers and one graph layer. `--disable_router`, `--disable_graph` and
`--disable_shared_specific` expose the required ablations. Shared/specific
orthogonality penalties are disabled by default (`alpha_shared=0`,
`alpha_specific=0`) because their coefficients must be calibrated on the
validation protocol.

## Locked configuration (A6, 2026-10-29)

`locked_configs.json` is the single source of truth for the frozen per-dataset
model. `scripts/run_locked.py` launches it:

```bash
PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python \
  scripts/run_locked.py --dataset FD002 --seed 0
PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python \
  scripts/run_locked.py --dataset DS08a --seed 0 --dry_run   # print command
```

| Dataset | Key settings | Params | Provenance |
|---|---|---|---|
| FD002/FD004 | A6 + patch_cnn, d64, cap=125, grouped scaler, fixed_cuts val, Adam recipe (41 ep) | 137,964 | `results_v3/fixed_cuts/A6` |
| DS03/DS08a | A6 + patch_cnn, d32/r2/h2, wd 1e-3, cond full, no cap, fixed_cuts val, batch 32 × 300 ep | 36,236 | `results_v3/ncmapss_tune/t3_d32_wd` (post-hoc) |

N-CMAPSS caveat: only 2 independent validation units, so the val_fc-selected
configs disagree with test; t3 is locked as the cross-dataset engineering
default and its N-CMAPSS test numbers are post-hoc until re-validated on fresh
seeds. Changing any of these settings means editing `locked_configs.json` and
its provenance — not passing different flags ad hoc.

## A1-A7 architecture presets

The follow-up experiments are exposed without changing the legacy default:

| Variant | Readout | Downsampling | Graph | Trend features |
|---|---|---|---|---|
| A1 | single | legacy | static | no |
| A2 | single | endpoint-aligned average | static | no |
| A3 | single | unconstrained learned causal filter | static | no |
| A4 | single | moment-constrained causal filter | static | no |
| A5 | single | moment-constrained causal filter | static | yes |
| A6 | single | moment-constrained causal filter | level dynamic | yes |
| A7 | single | moment-constrained causal filter | trend/reliability dynamic | yes |

All presets retain one patch-CNN stem, one cross-scale router, one sensor graph
and one scalar head. Use `--encoder patch_cnn` explicitly when running these
experiments. `--downsample_mode`, `--graph_mode`, `--readout` and
`--condition_mode` are also available independently for controlled ablations;
`--variant` takes precedence over those three structural settings.

## Data protocol

FD002/FD004 use 14 non-constant sensors, 3 operating settings and 50-cycle
windows by default. Train and validation are split by engine before windows
are created. The scaler is fitted only on training engines. Test evaluation
uses one final window per test engine.

N-CMAPSS is read in HDF5 chunks. Rows are first grouped by `(unit, cycle)` and
each cycle is represented by the mean of its physical sensors and operating
conditions. `T_*` is deliberately excluded: in these files `T_var` contains
degradation/health parameters, not a time or phase coordinate, so using it as
an input would leak latent health information. The default split is 7/2 dev
units for train/validation and all official test units remain locked until the
final evaluation. `--include_virtual_sensors` adds `X_v` as a separate input
ablation.

The N-CMAPSS adapter does not treat high-frequency rows as independent RUL
examples. It reports one final window per test unit. The adapter currently uses
cycle-level means; a phase-profile branch can be added only after a separate
protocol decision about within-cycle sampling.

## Run

From this directory, or with `PYTHONPATH=/home/v2/code`:

```bash
PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m rul_fusion.cli \
  --dataset FD002 \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --history 50 --rul_cap 0 --encoder patch_cnn --variant A1 \
  --seed 0 --output_dir results

PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m rul_fusion.cli \
  --dataset FD002 \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --history 50 --rul_cap 0 --encoder patch_cnn --variant A7 \
  --seed 0 --output_dir results

PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m rul_fusion.cli \
  --dataset DS03 \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --history 16 --seed 0 --output_dir results
```

Each run writes `best.pt`, `config.json`, `metrics.json`, `history.csv` and
`predictions.csv` under `results/<dataset>/seed_<seed>/`. Training logs keep
the DisMS-TS field order but defer test loss:

```text
epoch:1, train_loss:..., val_loss:..., test_loss:deferred
MAE: ... RMSE: ... NASA_Score: ...
```

The deferred test field prevents test-set checkpoint selection. `metrics.json`
contains raw-unit MAE, RMSE, NASA score, best epoch and split metadata.

## Unified-protocol baseline (FC-STGNN)

`--model fc_stgnn` trains the unmodified FC-STGNN model (loaded from
`--fc_stgnn_root`, default `/home/v2/FCSTGNN/FC_STGNN`) inside this runner, so
the baseline shares the engine-grouped split, train-only scaler,
validation-based checkpoint selection and single frozen test evaluation with
the fusion model. The reference FC-STGNN protocol is deliberately dropped:
window-level validation folds, per-condition MinMax scaling, label/125
normalization and best-epoch selection by test RMSE. Its per-dataset
architecture hyperparameters from `main_RUL.py::args_config` are kept verbatim
(FD002/FD004 only). The upstream code constructs on CUDA (`PositionalEncoding`
calls `.cuda()`), so this baseline requires a GPU. The last incomplete
training batch is dropped automatically because the model contains
BatchNorm1d. The wrapper ignores the condition input, matching the reference
training loop, which never fed its prepared operating-condition windows to the
network.

```bash
bash /home/v2/code/run_fcstgnn.sh            # cap=125, grouped scaler, seeds 0 1 2
RUL_CAP=0 bash /home/v2/code/run_fcstgnn.sh  # raw-label arm
```

## Verification

```bash
PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m pytest -q /home/v2/code/tests
PYTHONPATH=/home/v2/code /root/anaconda3/envs/torch23/bin/python -m compileall -q /home/v2/code/rul_fusion
```

The implementation is a reproducible runner and data adapter. It contains no
formal SOTA result; SOTA comparison must rerun matched splits, caps, windows,
features, seeds and metrics in the same runner.

## Chapter 4: quantile and conformal RUL

The fourth-chapter branch keeps the A6 representation and adds a non-crossing
q0.1/q0.5/q0.9 head. Training, early stopping and conformal calibration use
disjoint engine/unit sets. `global` is ordinary CQR; `hierarchical` adds
condition, predicted-stage and observable missingness groups with a global
fallback for sparse groups. The official test endpoint is evaluated only once
after the checkpoint and calibrator are frozen.

```bash
PYTHONPATH=/home/v2/.worktrees/ch4-hcq-cqr/code \
/root/anaconda3/envs/torch23/bin/python -m rul_fusion.quantile_cli \
  --dataset FD002 \
  --data_root /home/new_work/ch3_mtk_dg_probabilistic_design-85d9365/dataset \
  --output_dir results_v4/uncertainty \
  --calibration_mode hierarchical --alpha 0.2 --seed 0
```

Each run writes quantiles, calibrated intervals, calibration scores, a
serialised calibrator, grouping state and a manifest-ready config under
`results_v4/uncertainty/<mode>/<dataset>/seed_<n>/`. N-CMAPSS has only one
calibration unit under the registered split and is therefore an external
small-sample check rather than a strong coverage guarantee.

### FD002/FD004 probability baselines

HNN and Gaussian Deep Ensemble follow the compact Inception architecture,
positive two-parameter Gaussian output, initialization, and Gaussian-NLL
training in the official BayesRUL implementation. The ensemble cycles members
across minibatches and selects the joint checkpoint by validation mixture NLL.
EDL-NIG follows the cited paper's five-layer MLP, 0.3 dropout, Adam, 50 epochs,
batch size 32, NIG evidential loss and min-max sensor scaling. The paper does
not specify hidden widths, learning rate, or evidence penalty coefficient;
defaults are recorded as reconstruction choices. It uses the sensor list
shown in Figure 4 and the final cycle's sensor vector, without operation
settings. For a fair chapter comparison, EDL-NIG shares the engine split and
validation/test protocol; this differs from the paper's reported 80/20 split
and ten-fold cross-validation. Outputs include predictive parameters,
aleatoric/epistemic variance, NLL, CRPS, intervals, predictions and provenance.
See `references/probabilistic_baselines.md` for paper details and deviations.

Print the registered non-HCQ baseline commands first:

```bash
PYTHONPATH=code /root/anaconda3/envs/torch23/bin/python \
  code/scripts/run_chapter4_baselines.py --dataset FD002 --seed 0
```

Add `--run` only after reviewing the generated commands. The matrix includes
A6-point, A6-Quantile with Global CQR outputs, HNN, EDL-NIG and a five-member
Deep Ensemble. It does not launch HCQ-CQR, allowing that run to be managed
separately. The raw q10/q50/q90 columns in the quantile run are the uncalibrated
A6-Quantile outputs; the `lower`/`upper` columns are the Global CQR result.

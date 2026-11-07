# N-CMAPSS LOSO preparation amendment

Date: 2026-11-06. This amendment supersedes the search split and execution
order in `ncmapss-loso-tuning-prereg-2026-11-05.md`. The earlier document and
all P0, Arm B, and dev-tuning results remain unchanged. No LOSO result has
been used to write this amendment.

## Scope and isolation

- DS03 and DS08a are analyzed separately, with seed 0 and fixed-cut endpoints
  at life fractions 0.25, 0.50, 0.75, and 0.90.
- Registered calibration units 4 and 7 are excluded from all new training,
  checkpoint selection, and LOSO validation. The search pool is exactly
  `[1, 2, 3, 5, 6, 8, 9]`. Seven folds each train on six of these units and
  evaluate the seventh. The final model may train on all seven and calibrate
  on the registered two units.
- Search and frozen-checkpoint diagnostics do not load or evaluate the test
  split. The earlier test results were already observed and cannot provide an
  independent confirmation of this amendment.
- A6 architecture, non-crossing q10/q50/q90 output, pinball training loss,
  and Window-Weighted CQR rule remain fixed for the pilot. Hidden activation
  variants are deferred; output Softplus is not an activation search axis.

## Stage A: frozen-checkpoint diagnostic

Evaluate the existing P0 and three dev-tuning A6-Quantile checkpoints once on
the original three validation units `[3, 5, 6]`. Record each endpoint's three
pinball terms, q50 signed and squared errors, raw q10-q90 coverage, width,
and interval score. Summarize by unit and life fraction as well as overall.
For one checkpoint, `mean(residual)^2 + var(residual)` is only a descriptive
decomposition of its observed MSE, not an estimator of training variance.
No calibration or test labels are used to choose a new setting.

## Stage B: controlled LOSO pilot

Use the legacy activation, dropout 0.1, weight decay 1e-3, history 16,
batch size 32, Adam, and the original pinball-plus-regularization loss.
Learning rate is the only varied training parameter: `{1e-3, 3e-4}`.
This is 2 configurations x 7 folds x 2 datasets = 28 training runs.
Train for at most 300 epochs and record held-out endpoint pinball mean,
q50 RMSE, raw 80% interval score, PICP, and MPIW at every epoch.

For each configuration, average each epoch's metric equally across the seven
held-out units. Compare the fixed budgets obtained from the earliest minimum
of a centered five-epoch moving average of (a) pinball mean, (b) q50 RMSE,
and (c) raw interval score on the same training trajectories. Report all
three budgets and all metrics at each budget. If a smoothed minimum lies in
the final ten epochs, mark that criterion as boundary-censored; no final
model is selected from that curve until an equally applied extended budget
is registered. These same folds serve selection, so the selected score is a
development diagnostic rather than an unbiased generalization estimate.

## Decision after the pilot

Stage A and B results will be reported in full before any activation or loss
search is registered. A later dated amendment must fix the next grid and
selection rule before any further training. A q50-weighted pinball loss may
be considered as a separate training objective; a dual-head model is outside
the unchanged A6-Quantile structure. Final test evaluation and baseline
retraining require their own frozen configuration record and same-split
comparison protocol. The existing G9 calibration grid is historical and its
prior test-based evidence must be disclosed.

## Prepared commands and artifacts

The real N-CMAPSS HDF5 files are required under the remote data root. Stage A
needs no GPU; Stage B uses CUDA. Run each dataset's diagnostic in a separate
command, then run the two-dataset pilot:

```bash
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
$PY scripts/run_ncmapss_frozen_diagnostics.py --dataset DS03 --data_root "$DATA" --device cpu
$PY scripts/run_ncmapss_frozen_diagnostics.py --dataset DS08a --data_root "$DATA" --device cpu
$PY scripts/run_ncmapss_loso_search.py --data_root "$DATA" --grid pilot --max_epochs 300 --device cuda
```

Stage A writes `endpoints.csv`, `summary.csv`, and `manifest.json` under
`results_v4/ncmapss_frozen_diagnostics/<dataset>/`. Stage B writes fold caches,
per-epoch curves, aggregate curves, `criterion_budgets_pilot.csv`, and
`selection_pilot.csv/json` under `results_v4/ncmapss_loso_search/`. A completed
fold is atomically saved and reused only when its data path, size, modification
time, configuration, epoch count, and relevant source hashes agree. Partial
`--folds` smoke runs are not final selections.

The final-arm runner is gated by a JSON selection file with `status="frozen"`
and a `datasets` entry per dataset containing `config` (activation, learning
rate, dropout, weight decay), `budget_epoch`, `criterion="pinball_mean"`,
`search_units=[1,2,3,5,6,8,9]`, `calibration_units=[4,7]`, and
`n_folds=7`, and `boundary_censored=false`. It also requires `--eval_test`;
no such frozen selection file is issued at this preparation stage. A later
dated amendment must specify the final choice before test access or baseline
retraining.

## Existing LSTM-prob protocol audit

The frozen DS03 LSTM-prob and A6-Point result files both use the registered
train units `[1,2,8,9]`, validation `[3,5,6]`, calibration `[4,7]`, history
16, unit-cycle mean aggregation, raw RUL labels, and the same four fixed cuts
per test unit. LSTM-prob concatenates the sensor and condition sequences in
each 16-step window; it does not consume the full lifetime trajectory. Its
Gaussian NLL training and validation-RMSE checkpoint choice differ from
A6-Point's point objective and A6-Quantile's pinball objective.

A read-only comparison of the two existing `predictions.csv` files found 24
matching `(unit_id, sample_id, target_rul)` records and recomputed DS03 RMSE
of 5.5810 for LSTM-prob versus 13.0243 for A6-Point. LSTM-prob had lower RMSE
on each of the six test units. This verifies the local endpoint comparison,
not an independent test of a future tuned method. The LSTM-prob implementation
is a local adaptation of the cited method, with about 1.87 million parameters;
the A6-Point checkpoint has about 36 thousand. These differences must remain
visible in later baseline comparisons.

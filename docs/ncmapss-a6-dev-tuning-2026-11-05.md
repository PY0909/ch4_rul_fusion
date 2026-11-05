# N-CMAPSS A6-Quantile dev-only tuning amendment

Date: 2026-11-05. This note follows the frozen Arm B preregistration in
`docs/ncmapss-calibration-expansion-prereg-2026-11-05.md`. It records the Arm B
outcome before any new training trial. Arm A, G9, and G17 all remain reported;
none is selected using the already observed test endpoints.

## Observed Arm B outcome

The registered split remains 4 train / 3 validation / 2 calibration dev units.
Each dataset also has 6 separate test units; the 24 fixed-cut test endpoints
are correlated within those units. The model sees unit-cycle means, with
held-out sequences of roughly 56-93 cycles. The displayed test-cut RUL labels
range from 5 to 69, despite the uncapped target protocol.

| Dataset | Cuts per calibration unit | Window-weighted threshold | PICP | MPIW | Interval Score |
|---|---:|---:|---:|---:|---:|
| DS03 Arm A | 4 | 16.6273 | 0.9167 | 69.0836 | 82.4542 |
| DS03 G9 | 9 | 10.9938 | 0.9167 | 57.8164 | 75.8819 |
| DS03 G17 | 17 | 11.9801 | 0.9167 | 59.7890 | 77.0326 |
| DS08a Arm A | 4 | 16.3764 | 0.9167 | 61.0627 | 64.0286 |
| DS08a G9 | 9 | 20.4114 | 0.9583 | 69.1326 | 69.9924 |
| DS08a G17 | 17 | 20.4114 | 0.9583 | 69.1326 | 69.9924 |

Source: remote `results_v4/ncmapss_cqr_expanded/{DS03,DS08a}/seed_0/arm_comparison.csv`.
The added cuts separate the window-weighted and block-max thresholds, but
improve interval efficiency only on DS03. They do not move coverage toward
0.8 on either dataset. This does not establish calibration-window count as
the main bottleneck: all 9/17 cuts still come from the same two units, and G9
changes cut positions as well as count. The original statement that a
max-weight atom necessarily makes the 8-window weighted threshold at least
the unweighted maximum is not generally true; threshold equality was observed
for these particular weights and scores.

The A6-Point checkpoints selected epochs 87/188 on DS03/DS08a. The relevant
A6-Quantile checkpoints selected epochs **26/5**, with validation q50 RMSE
17.8745/16.3083. Training loss continued to decrease after those epochs while
validation q50 RMSE rose. The point and quantile checkpoints have different
losses and must not be conflated. DS03 q50 test RMSE is 19.8113 versus
13.0243 for A6-Point, so the next diagnostic focus is quantile training.

## Frozen dev-only training matrix

The A6 architecture (d_model 32, 2 heads, 2 routers, history 16), q10/q50/q90
head, alpha 0.2, registered engine split, fixed-cut endpoints, and
Window-Weighted CQR rule remain unchanged. Seed 0 is fixed. Each candidate
changes one training hyperparameter relative to P0:

| ID | Learning rate | Weight decay | Dropout |
|---|---:|---:|---:|
| P0, existing | 0.001 | 0.001 | 0.1 |
| lr3e4 | 0.0003 | 0.001 | 0.1 |
| wd3e3 | 0.001 | 0.003 | 0.1 |
| dropout02 | 0.001 | 0.001 | 0.2 |

Run all three new candidates on both datasets. Keep P0 as the recorded
reference; do not retrain or overwrite it. Checkpoints are chosen by the same
validation-endpoint q50 RMSE used in P0. Report every candidate's best epoch,
validation q50 RMSE, and calibration-only threshold. The validation set has
only three independent units and is also used for early stopping, so these
numbers are tuning diagnostics, not an unbiased performance estimate. No
test prediction or test-based candidate ranking belongs to this stage.

The script defaults to printing commands and refuses `--run` without CUDA:

```bash
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
$PY scripts/run_ncmapss_a6_dev_tuning.py --data_root "$DATA"
# After switching the server to GPU mode:
$PY scripts/run_ncmapss_a6_dev_tuning.py --data_root "$DATA" --run
$PY scripts/run_ncmapss_a6_dev_tuning.py --summarize
```

`--run` always passes `--defer_test`; it produces only calibration-only
artifacts under `results_v4/ncmapss_a6_dev_tuning/`. A failed or partial run
must be inspected before rerunning that directory. Further history-length or
split changes require a separate dev-only registration. The already touched
DS03/DS08a test endpoints cannot serve as independent confirmation of a
tuned configuration; any later test presentation must disclose this history.

## Executed dev-only results

All six new runs completed on 2026-11-05 with seed 0. The table reports the
validation-endpoint Window-Weighted CQR row; `P0` is the existing checkpoint.
All six new `calibration_only.json` files have `test_evaluated=false`.

| Dataset | ID | Best epoch | Validation q50 RMSE | Threshold | Validation PICP | Validation MPIW | Validation Interval Score |
|---|---|---:|---:|---:|---:|---:|---:|
| DS03 | P0 | 26 | 17.8745 | 16.6273 | 1.0000 | 70.4450 | 70.4450 |
| DS03 | lr3e4 | 221 | 16.1921 | 30.5314 | 1.0000 | 76.5412 | 76.5412 |
| DS03 | wd3e3 | 25 | 18.1195 | 19.7145 | 1.0000 | 74.7292 | 74.7292 |
| DS03 | dropout02 | 24 | 18.0103 | 20.0513 | 1.0000 | 75.7494 | 75.7494 |
| DS08a | P0 | 5 | 16.3083 | 16.3764 | 0.9167 | 60.9043 | 64.3147 |
| DS08a | lr3e4 | 15 | 16.0188 | 16.5210 | 0.9167 | 55.5213 | 58.4100 |
| DS08a | wd3e3 | 5 | 16.1822 | 17.8054 | 0.9167 | 56.0260 | 60.2280 |
| DS08a | dropout02 | 228 | 15.6610 | 31.4787 | 1.0000 | 71.4006 | 71.4006 |

Source: `results_v4/ncmapss_a6_dev_tuning/dev_summary.csv`, each run's
`cqr_variants_validation.csv`, and the P0 files in
`results_v4/ncmapss_comparison/cqr/`. On these six runs, the three CQR
variants have identical validation thresholds and metrics, so this trial
does not identify a benefit from weighting.

The lower learning rate reduces q50 RMSE on both datasets, but DS03's
calibration threshold rises by 13.90 and its validation interval widens by
6.10. On DS08a, the same setting lowers both q50 RMSE and interval width.
Dropout 0.2 gives the lowest DS08a q50 RMSE while nearly doubling its
threshold and widening its interval by 10.50. Weight decay 0.003 does not
improve DS03 q50 RMSE. These are diagnostics on three validation units per
dataset, which also selected checkpoints; they do not establish a general
ranking or justify replacing the frozen P0 result. No new test endpoint was
evaluated during this matrix.

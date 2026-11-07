# N-CMAPSS LOSO budget comparison amendment

Date: 2026-11-07. This prospective development-stage amendment follows the
observed 300-epoch LOSO pilot. It does not retroactively preregister that pilot
or make its seven-fold scores independent estimates. No new test evaluation is
authorized by this document.

## Question and fixed protocol

Compare the pinball-mean and raw 80% interval-score budget rules under a
120-epoch cap and patience 30. DS03 and DS08a are separate analyses. Keep
seed 0, history 16, batch size 32, A6-Quantile's architecture and equal-weight
pinball-plus-regularization training loss, legacy activation, dropout 0.1,
weight decay 1e-3, and Adam. Learning rate is either 1e-3 or 3e-4. The
Window-Weighted CQR rule is unchanged and is not fitted in this comparison.

Registered calibration engines [4, 7] remain untouched. The search pool is
[1, 2, 3, 5, 6, 8, 9]; in each of seven folds, train on six and evaluate the
seventh at the four fixed-cut validation endpoints. The test HDF5 split is not
loaded. Each fold is a development validation fold, not an independent test.

## Epoch and stopping rule

At every epoch record held-out pinball mean, q50 RMSE, raw interval score,
raw PICP, and raw MPIW. For both pinball mean and raw interval score, use the
earliest minimum of a centered five-epoch moving average. A new strict
minimum resets that metric's patience counter. The latest available centered
epoch is current epoch minus two. Stop a fold when neither metric has achieved
a new strict minimum for 30 successive available centers, or after epoch 120,
whichever comes first. q50 RMSE is diagnostic and does not stop training.

Both budget rules select epoch records from the same trajectory. Record each
fold's observed metrics at each selected epoch. Compare rules by averaging
the seven held-out engines equally; report every engine, selected epoch,
stop epoch, and coverage. Report the median selected epoch as a development
candidate for a later fixed-budget seven-engine training run. A selected
epoch in the final ten epochs when training reaches the 120-epoch cap is
marked boundary censored. No rule, learning rate, or final test arm is frozen
by this run.

The raw interval score equals ten times the sum of q10 and q90 pinball terms
at alpha 0.2, whereas pinball mean also includes q50. This comparison tests
the observed median-versus-tail tradeoff. Raw interval quality is a proxy,
not a measurement of calibrated Window-Weighted CQR quality. A subsequent
calibrated comparison would need separate development calibration engines.

## Outputs and gate

The runner writes per-epoch fold caches, `fold_budgets.csv`,
`comparison.csv`, and `manifest.json` under
`results_v4/ncmapss_loso_budget_compare/`. The manifest must identify the
complete seven-fold run before any comparison is interpreted. It must record
source and data identities, all training and stopping settings, and that test
was not loaded. Partial synthetic or smoke runs cannot become selections.

The real-data command, after code verification and AutoDL startup, is:

```bash
/root/miniconda3/envs/kaf_profiti/bin/python -u \
  scripts/run_ncmapss_loso_budget_compare.py \
  --data_root /root/autodl-tmp/kst_light_v2/dataset --device cuda
```

Do not run `run_ncmapss_loso_final.py` from these outputs. Final fixed-budget
training, G9 calibration, baseline retraining, and any additional descriptive
test evaluation require a later frozen protocol record.

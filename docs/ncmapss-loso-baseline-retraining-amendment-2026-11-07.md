# N-CMAPSS LOSO baseline retraining amendment

Date: 2026-11-07. Written before the new baseline LOSO search. The existing
P0, Arm B, and A6 pilot test endpoints have already been observed; any later
test comparison is a descriptive recheck, not independent external evidence.

## Shared protocol

Run DS03 and DS08a separately with seed 0, sensor history 16, raw flight RUL,
unit-cycle mean aggregation, and a z-score scaler fitted on training engines
only. Reserve calibration engines `[4, 7]`. The development pool is exactly
`[1, 2, 3, 5, 6, 8, 9]`; in each of seven folds, train on six engines and
evaluate the held-out engine's four fixed-cut endpoints at life fractions
`[0.25, 0.5, 0.75, 0.9]`. Neither calibration labels nor test data are loaded
during selection. All eight baselines use the same split and endpoints.

The baselines are A6-Point, FC-STGNN, HNN, EDL-NIG, LSTM-prob,
TCN-Transformer-MC, MSAUQ, and BCPT. Their model structures and native losses
stay as implemented in the P0 comparison. In particular, A6-Quantile and
Window-Weighted CQR are outside this baseline runner and remain unchanged.
The NC optimizer settings from P0 remain fixed: Adam, learning rate `1e-3`,
batch size 32, weight decay `0` for both point models and EDL-NIG, and `1e-3`
for the other probabilistic models. Method-specific dropout, MC draw count,
and BCPT parameters retain their P0 values.

## Selection and budget

Each fold trains for at most 120 epochs with patience 30. Select the earliest
strict minimum of the method's registered held-out endpoint score: RMSE for
A6-Point, FC-STGNN, EDL-NIG, LSTM-prob, TCN-Transformer-MC, and MSAUQ; NLL
for HNN; fused RMSE for BCPT. BCPT's validation draw is computed after each
epoch's parameter update so the score matches the saved checkpoint. Persist
the best checkpoint and full epoch history. Stop after 30 epochs without a new
strict minimum. The seven selected
epochs yield a candidate fixed budget by their median. Report each held-out
engine's RMSE at its selected checkpoint and the equal-engine mean RMSE as a
common descriptive comparison. Never rank the methods by their different
native selection-score scales. A partial smoke cannot freeze a budget.

Review the complete seven-fold outputs and A6 budget-comparison outputs before
writing a separate JSON with `status: "frozen"`. For each dataset it must name
the seven search units, two calibration units, `n_folds: 7`, and each selected
baseline's `epochs` (1..120). The final runner checks the whole requested
selection before opening test. It then trains on all seven search engines for
exactly the frozen number of epochs, without validation or early stopping.
There is no test-based hyperparameter selection. Test evaluation requires the
explicit `--eval_test` flag. Point RMSE/MAE and probabilistic native interval
metrics are reported separately; the baseline native intervals are not
Window-Weighted CQR intervals. The previous test observations and this new
touch must be disclosed in the thesis.

## Code and provenance

Runner: `scripts/run_ncmapss_loso_baselines.py`. It writes one checkpoint,
history, and manifest per fold; completed runs are reused only when source
hashes, HDF5 identity, split settings, and training settings match. Aggregate
tables are `development.csv`, `development_summary.csv`, and `final.csv`.
Final outputs also include predictions and metrics per model.

FC-STGNN source is the authors' repository
`https://github.com/Frank-Wang-oss/FCSTGNN`, cloned locally at commit
`93f2387feff4457b17016b33931a7dc6ec757719`. The runner reads its
unmodified `Model.py` and `Model_Base.py`, records their hashes, and accepts
`--fc_source` for the server location. The upstream source hard-codes CUDA
allocations; a temporary CPU-only compatibility context is used for local
smoke tests, while GPU runs execute the original source directly.

After AutoDL starts, run A6's already registered budget comparison first.
Then run the baseline development matrix with the remote data and FC source:

```bash
PY=/root/miniconda3/envs/kaf_profiti/bin/python
DATA=/root/autodl-tmp/kst_light_v2/dataset
$PY scripts/run_ncmapss_loso_baselines.py \
  --phase development --data_root "$DATA" --device cuda \
  --fc_source /home/v2/FCSTGNN/FC_STGNN
```

Do not run the final phase until the two complete development results have
been reviewed and the configuration has been frozen. No final test run is
authorized merely by this amendment.

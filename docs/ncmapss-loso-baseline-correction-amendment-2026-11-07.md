# N-CMAPSS LOSO baseline correction amendment

Date: 2026-11-07, written before any rerun. It amends the configuration part
of `ncmapss-loso-baseline-retraining-amendment-2026-11-07.md`; splits,
endpoints, selection metrics, the 120-epoch cap with patience 30, and the
final-arm gate stay as written there. No test data or calibration labels are
loaded by anything below.

Revision note: an earlier draft of this amendment (commit 3a639f4) proposed a
400-epoch cap with patience 100 for the baselines only. That proposal was
withdrawn on the same day, before any run, by the user's decision that every
compared model, the A6-Quantile main model included, uses one common
development budget of 120 epochs with patience 30.

## Finding 1: A6-Point configuration did not match P0

`_point_model` in `scripts/run_ncmapss_loso_baselines.py` built A6-Point as
d_model 64 / 4 heads / 4 routers (the FD lock) and `_train_point` used Adam
without weight decay for both point models. The P0 N-CMAPSS A6-Point
(`results_v4/ncmapss_comparison/point/A6/*/seed_0/config.json`) is d_model 32
/ 2 heads / 2 routers with weight decay 1e-3, and the LOSO A6-Quantile also
uses d32/2/2. FC-STGNN's P0 weight decay is 0, which the runner already
matched.

**Correction**: A6-Point = d32 / 2 heads / 2 routers / dropout 0.1, Adam
lr 1e-3, weight decay 1e-3; FC-STGNN unchanged (weight decay 0).

## Finding 2: LSTM-prob does not converge under the common budget

Audit result: the LOSO LSTM-prob uses the same settings as its P0 N-CMAPSS run
(Adam lr 1e-3, weight decay 1e-3, dropout 0.03, batch 32, validation-RMSE
selection, Gaussian NLL loss); only the training budget differs.

- P0 DS03 history (`results_v4/ncmapss_comparison/probabilistic/DS03/
  lstm_prob/seed_0/history.csv`): training NLL stays near 4.50 and
  validation RMSE near 20 through about epoch 224, then the mean starts to fit
  (validation RMSE 14.30 at epoch 249, 8.25 at 274, 7.58 at 299, still falling
  at the 300-epoch limit).
- LOSO under 120 epochs with patience 30: 13 of 14 folds selected epoch 8 or
  earlier with held-out RMSE about 12–24, consistent with a near-constant
  predictor; the remaining fold (DS03, held unit 5) selected epoch 119.

Conclusion: under the common budget, LSTM-prob stays inside its NLL plateau.
This is not an implementation error, and no method-specific budget is
granted. LSTM-prob is reported as **not converged under the common 120/30
budget**, with the P0 300-epoch curve disclosed as the reason. The P0 DS03
LSTM-prob test RMSE (5.58) came from a run still improving at its 300-epoch
limit; its plateau-escape timing is a disclosed single-seed sensitivity.

## Common budget and truncation reporting

All compared models — the A6-Quantile budget comparison and all eight
baselines (A6-Point, FC-STGNN, HNN, EDL-NIG, LSTM-prob, TCN-Transformer-MC,
MSAUQ, BCPT) — use at most 120 epochs with patience 30 (earliest strict
minimum of each method's registered score).

Every development fold now records its stop epoch and stop reason (`cap` or
`patience`) in addition to the existing `boundary_censored` flag (selected
epoch later than cap − 10). Runs stopped by the cap are disclosed per method.
In the existing A6-Quantile budget comparison, 2 of 28 fold runs were stopped
by the cap: DS03 lr 3e-4 held unit 3 (selected epoch 108) and DS08a lr 3e-4
held unit 1 (selected epoch 90); neither is boundary censored under the
cap − 10 rule. They are reported as cap-stopped, and the A6-Quantile
comparison is not rerun.

## Reruns and retained records

- Rerun: the eight baselines × 7 folds × 2 datasets at 120/30 with the
  corrected A6-Point into `results_v4/ncmapss_loso_baselines_v2/`
  (`scripts/queue_ncmapss_loso_baselines.sh`). The runner's result identity
  includes its source hash, so all eight rerun under one code version; the
  seven baselines whose configuration did not change are expected to
  reproduce their previous development results up to GPU nondeterminism, which
  serves as a consistency check.
- Superseded but retained unchanged: `results_v4/ncmapss_loso_baselines/`
  (FD-sized A6-Point). It must not be used to freeze any budget.
- The candidate fixed budget stays the low median of the seven selected
  epochs. Budgets are frozen only in the next step, after this rerun is
  reviewed; that step also records the A6 selection criterion.

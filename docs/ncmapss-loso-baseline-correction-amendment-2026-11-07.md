# N-CMAPSS LOSO baseline correction amendment

Date: 2026-11-07, written before any rerun. It amends the configuration and
budget parts of `ncmapss-loso-baseline-retraining-amendment-2026-11-07.md`;
splits, endpoints, selection metrics and the final-arm gate stay as written
there. No test data or calibration labels are loaded by anything below.

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

## Finding 2: LSTM-prob early selection is a budget artifact

Audit result: the LOSO LSTM-prob uses the same settings as its P0 N-CMAPSS run
(Adam lr 1e-3, weight decay 1e-3, dropout 0.03, batch 32, validation-RMSE
selection, Gaussian NLL loss); only the training budget differs.

- P0 DS03 history (`results_v4/ncmapss_comparison/probabilistic/DS03/
  lstm_prob/seed_0/history.csv`): training NLL stays near 4.50 and
  validation RMSE near 20 through about epoch 224, then the mean starts to fit
  (validation RMSE 14.30 at epoch 249, 8.25 at 274, 7.58 at 299, still falling
  at the 300-epoch limit).
- LOSO under the 120-epoch cap with patience 30: 13 of 14 folds selected
  epoch 8 or earlier with held-out RMSE about 12–24, consistent with a
  near-constant predictor; the remaining fold (DS03, held unit 5) selected
  epoch 119 at the cap.

Conclusion: the 120/30 rule stops LSTM-prob inside its NLL plateau. This is a
selection-protocol artifact, not an implementation error. It also means the
P0 DS03 LSTM-prob test RMSE (5.58) came from a run that was still improving
at its budget limit; with one seed, the plateau-escape timing is a source of
sensitivity that must be disclosed.

## Correction 2: one uniform development budget for all eight baselines

All eight baselines (A6-Point, FC-STGNN, HNN, EDL-NIG, LSTM-prob,
TCN-Transformer-MC, MSAUQ, BCPT) rerun the seven-fold development search with
a **400-epoch cap and patience 100** (earliest strict minimum of each
method's registered score; a new strict minimum resets patience). No
method-specific exception is allowed. Rationale: the P0 plateau ended near
epoch 225, and on its 25-epoch sampled grid new strict minima occurred at
intervals of at most about 75 epochs before the escape. A selected epoch
later than cap − 10 is flagged `boundary_censored`; no budget may be frozen
for a censored method until a further amendment applies an extended cap to
every method.

## Reruns, retained records, unaffected arms

- Rerun: 8 baselines × 7 folds × 2 datasets into
  `results_v4/ncmapss_loso_baselines_v2/` (`scripts/queue_ncmapss_loso_baselines.sh`).
- Superseded but retained unchanged: `results_v4/ncmapss_loso_baselines/`
  (120/30 run with the FD-sized A6-Point). It must not be used to freeze any
  budget.
- Unaffected: the A6-Quantile budget comparison
  (`results_v4/ncmapss_loso_budget_compare/`). Its 300-epoch pilot preceded the
  120/30 choice, all its selected budgets lie at or below 108 epochs, and none
  is boundary censored, so it is not rerun.
- The candidate fixed budget stays the low median of the seven selected
  epochs. Budgets are frozen only in the next step, after this rerun is
  reviewed; that step also records the A6 selection criterion.

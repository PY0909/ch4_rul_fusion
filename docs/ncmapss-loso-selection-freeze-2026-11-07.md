# N-CMAPSS LOSO selection freeze

Date: 2026-11-07. Written after all development outputs listed below were
complete and before any final-arm training or test access under this
protocol. Frozen record: `results_v4/ncmapss_loso_frozen/selection.json`
(status `frozen`). The P0, Arm B and dev-tuning test observations predate this
protocol and remain disclosed; any final-arm test number is a descriptive
recheck, not independent confirmation.

Inputs: `results_v4/ncmapss_loso_budget_compare/comparison.csv` and
`fold_budgets.csv` (A6-Quantile, 7 folds × 2 learning rates × 2 datasets,
120/30), and `results_v4/ncmapss_loso_baselines_corrected/` (eight
baselines, A6-Point at the P0 lock, 120/30). Search units
`[1, 2, 3, 5, 6, 8, 9]`; calibration units `[4, 7]` untouched throughout.

## A6-Quantile: selection criterion and frozen configuration

Criterion: **development raw 80% interval score** of the q10/q90 head output,
equal-weighted over the seven held-out engines, at each configuration's budget
(low median of its seven per-fold raw-IS-selected epochs).

Rationale: Chapter 4's primary target is interval quality. At α = 0.2 the raw
interval score equals exactly 10 × (q10 pinball + q90 pinball), i.e. the
interval-endpoint part of the pinball sum without the median term. It is a
**pre-CQR proxy**: it scores the uncalibrated quantile head, not the
Window-Weighted CQR interval, and must not be described as an optimized
calibrated interval score.

Disclosure of timing: this criterion replaces the pinball-mean primary named in
`ncmapss-loso-tuning-prereg-2026-11-05.md` and is chosen after the budget
comparison was observed.

| Dataset | Frozen config | Budget | Fold range | Dev raw IS | Dev raw PICP | Dev q50 RMSE | Dev pinball |
|---|---|---:|---|---:|---:|---:|---:|
| DS03 | legacy, lr 3e-4, dropout 0.1, wd 1e-3 | 27 | 22–108 | 50.72 | 0.607 | 15.66 | 3.954 |
| DS08a | legacy, lr 1e-3, dropout 0.1, wd 1e-3 | 22 | 9–49 | 56.45 | 0.607 | 17.01 | 4.396 |

Sensitivity (development only, never evaluated on test): under the
pinball-mean rule, DS03 selects the same configuration and budget (lr 3e-4,
27 epochs; pinball 3.901, raw IS 51.27). **DS08a differs**: pinball-mean selects
lr 3e-4 with 35 epochs (pinball 4.365 vs 4.469; raw IS 61.16 vs 56.45). The
frozen raw-IS choice therefore trades some DS08a median accuracy (development
q50 RMSE 17.01 vs 16.32) for tail quality. Both gaps are small relative to
seven-fold variability; the configuration is "selected by the registered
rule", not "significantly better".

Cap stops: the DS03 frozen configuration has one fold (held unit 3) whose
training stopped at the 120 cap with selected epoch 108; the DS08a frozen
configuration has none. No A6 selection is boundary censored.

## Baselines: frozen epochs

Low median of each method's seven selected epochs under its registered
selection score (RMSE for A6-Point, FC-STGNN, EDL-NIG, LSTM-prob,
TCN-Transformer-MC, MSAUQ; NLL for HNN; fused RMSE for BCPT).

| Method | DS03 epochs | DS08a epochs | Cap-stopped folds (DS03/DS08a) | Boundary-censored folds (DS03/DS08a) |
|---|---:|---:|---|---|
| A6-Point (P0 lock) | 49 | 18 | 2 / 0 | 0 / 0 |
| FC-STGNN | 63 | 34 | 2 / 0 | 2 / 0 |
| HNN | 44 | 46 | 1 / 1 | 0 / 0 |
| EDL-NIG | 78 | 31 | 2 / 1 | 1 / 0 |
| LSTM-prob | 6 | 5 | 1 / 0 | 1 / 0 |
| TCN-Transformer-MC | 47 | 37 | 2 / 0 | 1 / 0 |
| MSAUQ | 25 | 43 | 0 / 0 | 0 / 0 |
| BCPT | 38 | 38 | 0 / 0 | 0 / 0 |

LSTM-prob is frozen at its development median and reported as **not converged
under the common 120/30 budget** (see the correction amendment). Cap-stopped and
boundary-censored folds are disclosed; the median is robust to a minority of
late folds, and no method receives a different budget.

## Final arm (next step), and its order of operations

1. **Training stage (no test access)**: every model trains on all seven
   search engines for exactly its frozen epochs, no validation, no early
   stopping, seed 0. A6-Quantile then predicts the G9 calibration endpoints of
   engines `[4, 7]`. Each run writes its checkpoint, history, calibration
   predictions (A6) and a manifest recording the selection file hash, frozen
   epochs, checkpoint SHA-256 and `test_loaded: false`.
2. **Verification**: configurations, epoch counts, units and checkpoint hashes
   are checked against this record before any test access.
3. **Evaluation stage (one descriptive recheck)**: requires `--eval_test`;
   loads each verified checkpoint (hash re-checked), predicts the six test
   engines' four fixed-cut endpoints once, fits Window-Weighted CQR weights on
   unlabeled test covariates as in production, and writes per-engine errors.
   Every access is logged in `test_touches.log`.

Training seven-engine models with budgets estimated on six-engine folds is a
small, disclosed shift. The G9 calibration grid has prior test history (Arm B),
also disclosed. Thesis tables keep LOSO development rankings, final descriptive
test comparisons, interval metrics and per-engine errors separate, and never
present differences between the P0 and LOSO protocols as gains.

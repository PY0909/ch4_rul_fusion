# Fourth-chapter uncertainty results

Each run is stored as `<calibration_mode>/<dataset>/seed_<seed>/`.

Required files are `best.pt`, `metrics.json`, `history.csv`,
`predictions.csv`, `calibration_scores.csv`, `calibrator.json`,
`grouping.json`, `config.json`, and `manifest.json`.

`global` is ordinary conformalized quantile regression. `hierarchical` adds
condition, predicted-stage, and observed-missingness groups and falls back to
the global threshold for sparse or unseen groups. The calibration endpoint
cuts are correlated within an engine/unit, so the runner retains one
conservative (maximum) nonconformity score per engine/unit before fitting CQR.
Test evaluation occurs only after checkpoint and calibrator selection.

The quantile head is cumulative and Softplus-parameterized: q10, q50, and q90
are ordered and non-negative. Each manifest records the train-only scaler
scope, code revision, test timestamp, and any test-only sensor corruption.

N-CMAPSS has very few independent dev units. Its interval metrics are
external diagnostics and must not be reported as population-level coverage
guarantees.

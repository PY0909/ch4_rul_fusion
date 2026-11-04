# Chapter 4 evidence map

This file is filled from committed result manifests after the registered
matrix runs. Every claim must point to a dataset, method, seed, and the saved
`metrics.json`/`predictions.csv` artifacts.

| Claim | Required evidence | Status |
|---|---|---|
| Quantile head produces ordered q10/q50/q90 | `test_uncertainty.py`, prediction files | implemented; training pending |
| Global CQR improves clean coverage | FD002/FD004 global runs, `metrics.json` | pending |
| HCQ-CQR improves a pre-registered subgroup | group metrics and unit bootstrap | pending |
| Sensor corruption changes coverage and width predictably | FD002 corruption runs | pending |
| N-CMAPSS external check | DS03/DS08a per-unit manifests | pending |

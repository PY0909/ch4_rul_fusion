import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.data import DataBundle, FeatureScaler, WindowDataset
from rul_fusion.model import MultiScalePhaseGraphRUL
from rul_fusion.runner import fit_and_evaluate


def _bundle():
    rng = np.random.default_rng(2)
    def samples(n, offset):
        return [(rng.normal(size=(8, 4)).astype("f4"),
                 rng.normal(size=(8, 2)).astype("f4"),
                 float(offset + i), 100 + i, i) for i in range(n)]
    return DataBundle(WindowDataset(samples(8, 1)), WindowDataset(samples(4, 2)),
                      WindowDataset(samples(2, 3)), FeatureScaler(np.zeros(6), np.ones(6)),
                      {"dataset": "SYNTH", "train_units": list(range(8)),
                       "val_units": list(range(4)), "test_units": [100, 101]}, 4, 2)


def test_runner_defers_test_and_writes_result_schema(tmp_path, capsys):
    bundle = _bundle()
    model = MultiScalePhaseGraphRUL(4, 2, d_model=16, num_scales=2,
                                    num_heads=4, num_routers=2)
    result = fit_and_evaluate(bundle, model, tmp_path, epochs=1, batch_size=2,
                              lr=1e-3, patience=2, device="cpu", seed=0)
    output = capsys.readouterr().out
    assert "test_loss:deferred" in output
    assert "MAE:" in output
    for name in ["metrics.json", "predictions.csv", "history.csv", "config.json", "best.pt"]:
        assert (tmp_path / name).exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(["MAE", "RMSE", "NASA_Score"]).issubset(metrics)
    assert result["best_epoch"] == 1

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.data import DataBundle, WindowDataset
from rul_fusion.model import MultiScalePhaseGraphRUL
from rul_fusion.quantile_runner import fit_quantile_and_calibrate
from rul_fusion.uncertainty import A6QuantileRUL


def _samples(count, history=8, sensors=4, conditions=2):
    rng = np.random.default_rng(3)
    result = []
    for index in range(count):
        x = rng.normal(size=(history, sensors)).astype(np.float32)
        w = rng.normal(size=(history, conditions)).astype(np.float32)
        y = float(20.0 - index * 0.2 + x[-1].mean())
        result.append((x, w, y, index % 4, index))
    return result


def test_quantile_runner_writes_recomputable_artifacts(tmp_path):
    bundle = DataBundle(
        train=WindowDataset(_samples(16)),
        val=WindowDataset(_samples(8)),
        test=WindowDataset(_samples(4)),
        scaler=None,
        metadata={"dataset": "DS03", "split_seed": 0},
        sensor_dim=4,
        condition_dim=2,
        val_endpoint=WindowDataset(_samples(4)),
        test_endpoint=None,
        calibration_endpoint=WindowDataset(_samples(8)),
    )
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=4, condition_dim=2, d_model=8, num_scales=3,
        num_heads=2, num_routers=2, encoder="patch_cnn", variant="A6",
    )
    model = A6QuantileRUL(backbone, dropout=0.0)
    result = fit_quantile_and_calibrate(
        bundle, model, tmp_path, epochs=2, batch_size=4, lr=1e-3,
        weight_decay=0.0, patience=2, alpha=0.2, calibration_mode="global",
        seed=0, device="cpu")
    assert np.isfinite(result["PICP"])
    assert "NASA_Score" not in result
    for filename in ("best.pt", "metrics.json", "history.csv", "predictions.csv",
                     "calibration_scores.csv", "calibrator.json", "grouping.json",
                     "config.json", "manifest.json"):
        assert (tmp_path / filename).exists()

import csv
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_data import _write_ncmapss_nine_units  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.uncertainty import A6QuantileRUL  # noqa: E402


def test_frozen_diagnostic_uses_validation_only(tmp_path):
    script = Path(__file__).parents[1] / "scripts/run_ncmapss_frozen_diagnostics.py"
    assert script.exists(), "frozen-checkpoint diagnostic entry point is missing"
    diagnostics = importlib.import_module("scripts.run_ncmapss_frozen_diagnostics")
    path = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(path)
    backbone = MultiScalePhaseGraphRUL(
        sensor_dim=14, condition_dim=4, d_model=32, num_heads=2,
        num_routers=2, dropout=0.1, encoder="patch_cnn", variant="A6",
        condition_mode="full")
    model = A6QuantileRUL(backbone, dropout=0.1)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    out = tmp_path / "diagnostics"
    diagnostics.main([
        "--dataset", "DS03", "--data_root", str(tmp_path),
        "--dev_path", str(path), "--test_path", str(tmp_path / "missing.h5"),
        "--checkpoint", f"p0={checkpoint}", "--output_dir", str(out),
        "--device", "cpu"])
    with (out / "DS03" / "endpoints.csv").open(newline="") as handle:
        endpoints = list(csv.DictReader(handle))
    with (out / "DS03" / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(endpoints) == 3 * 4
    assert {int(row["unit_id"]) for row in endpoints} == {3, 5, 6}
    assert {float(row["life_fraction"]) for row in endpoints} == {
        0.25, 0.5, 0.75, 0.9}
    assert all(float(row["raw_width"]) >= 0 for row in endpoints)
    assert {row["group_kind"] for row in summary} == {
        "overall", "unit", "life_fraction"}
    assert next(row for row in summary if row["group_kind"] == "overall")[
        "n_endpoints"] == "12"

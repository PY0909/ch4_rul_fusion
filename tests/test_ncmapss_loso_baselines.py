import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_data import _write_ncmapss_nine_units
from scripts.run_ncmapss_loso_baselines import main


def test_development_fold_excludes_calibration_and_never_loads_test(tmp_path):
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    out = tmp_path / "runs"
    args = ["--phase", "development", "--data_root", str(tmp_path),
            "--datasets", "DS03", "--methods", "a6_point", "lstm_prob",
            "--dev_path", str(dev), "--test_path", str(tmp_path / "absent.h5"),
            "--max_epochs", "1", "--patience", "1", "--folds", "1",
            "--output_dir", str(out), "--device", "cpu"]
    main(args)
    rows = list(csv.DictReader((out / "development.csv").open()))
    assert {row["method"] for row in rows} == {"a6_point", "lstm_prob"}
    assert all(row["held_unit"] == "1" for row in rows)
    assert all(float(row["val_rmse"]) >= 0 for row in rows)
    for row in rows:
        run = out / "folds" / "DS03" / row["method"] / "held_1"
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["train_units"] == [2, 3, 5, 6, 8, 9]
        assert manifest["val_units"] == [1]
        assert manifest["calibration_units_excluded"] == [4, 7]
        assert manifest["test_loaded"] is False
        assert (run / "best.pt").exists()
        assert not (run / "predictions.csv").exists()
    main(args)
    assert len(list(csv.DictReader((out / "development.csv").open()))) == 2


def test_final_requires_frozen_selection_and_explicit_test_gate(tmp_path, capsys):
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "candidate", "datasets": {"DS03": {"methods": {
            "a6_point": {"epochs": 1}, "lstm_prob": {"epochs": 1}}}}}))
    args = ["--phase", "final", "--data_root", str(tmp_path),
            "--datasets", "DS03", "--methods", "a6_point", "lstm_prob",
            "--dev_path", str(dev), "--test_path", str(dev),
            "--selection", str(selection), "--output_dir", str(tmp_path / "final"),
            "--device", "cpu"]
    with pytest.raises(SystemExit):
        main(args)
    assert "--eval_test" in capsys.readouterr().err
    with pytest.raises(ValueError, match="frozen"):
        main(args + ["--eval_test"])
    data = json.loads(selection.read_text())
    data["status"] = "frozen"
    data["datasets"]["DS03"].update({
        "search_units": [1, 2, 3, 5, 6, 8, 9],
        "calibration_units": [4, 7], "n_folds": 7})
    selection.write_text(json.dumps(data))
    main(args + ["--eval_test"])
    run = tmp_path / "final" / "DS03" / "a6_point"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["train_units"] == [1, 2, 3, 5, 6, 8, 9]
    assert manifest["val_units"] == []
    assert manifest["calibration_units"] == [4, 7]
    assert manifest["test_evaluated"] is True
    assert (run / "predictions.csv").exists()
    prob = tmp_path / "final" / "DS03" / "lstm_prob"
    assert (prob / "predictions.csv").exists()
    assert json.loads((prob / "manifest.json").read_text())["val_units"] == []


def test_final_prevalidates_every_selected_method_before_test(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "frozen", "datasets": {"DS03": {
            "search_units": [1, 2, 3, 5, 6, 8, 9],
            "calibration_units": [4, 7], "n_folds": 7,
            "methods": {"a6_point": {"epochs": 1}}}}}))
    out = tmp_path / "final"
    with pytest.raises(ValueError, match="lstm_prob"):
        main(["--phase", "final", "--data_root", str(tmp_path),
              "--datasets", "DS03", "--methods", "a6_point", "lstm_prob",
              "--selection", str(selection), "--eval_test",
              "--output_dir", str(out), "--device", "cpu"])
    assert not out.exists()


def test_fc_stgnn_loads_real_local_source_and_runs_one_fold(tmp_path):
    source = Path("/Users/ppy/研/FCSTGNN-upstream/FC_STGNN")
    if not (source / "Model.py").exists():
        pytest.skip("FC-STGNN source is unavailable")
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    out = tmp_path / "runs"
    main(["--phase", "development", "--data_root", str(tmp_path),
          "--datasets", "DS03", "--methods", "fc_stgnn",
          "--dev_path", str(dev), "--test_path", str(tmp_path / "absent.h5"),
          "--fc_source", str(source), "--max_epochs", "1", "--patience", "1",
          "--folds", "1", "--output_dir", str(out), "--device", "cpu"])
    assert (out / "folds/DS03/fc_stgnn/held_1/best.pt").exists()


def test_all_eight_baselines_short_development_matrix(tmp_path):
    source = Path("/Users/ppy/研/FCSTGNN-upstream/FC_STGNN")
    if not (source / "Model.py").exists():
        pytest.skip("FC-STGNN source is unavailable")
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    out = tmp_path / "matrix"
    main(["--phase", "development", "--data_root", str(tmp_path),
          "--datasets", "DS03", "DS08a", "--dev_path", str(dev),
          "--test_path", str(tmp_path / "absent.h5"), "--fc_source", str(source),
          "--max_epochs", "1", "--patience", "1", "--folds", "1",
          "--mc_samples", "3", "--output_dir", str(out), "--device", "cpu"])
    rows = list(csv.DictReader((out / "development.csv").open()))
    assert len(rows) == 16
    assert all(float(row["val_rmse"]) >= 0 for row in rows)
    summary = list(csv.DictReader((out / "development_summary.csv").open()))
    assert len(summary) == 16
    assert all(float(row["mean_held_rmse"]) >= 0 for row in summary)
    assert all(row["candidate_fixed_epochs"] == "" for row in summary)
    assert json.loads((out / "manifest.json").read_text())["status"] == "partial_smoke"


def test_a6_point_matches_p0_lock_and_uniform_budget(tmp_path):
    import types
    from scripts import run_ncmapss_loso_baselines as runner
    assert runner.A6_POINT_DIMS == {"d_model": 32, "num_heads": 2, "num_routers": 2}
    assert runner.POINT_WEIGHT_DECAY == {"a6_point": 1e-3, "fc_stgnn": 0.0}
    assert (runner.DEV_MAX_EPOCHS, runner.DEV_PATIENCE) == (120, 30)
    bundle = types.SimpleNamespace(sensor_dim=14, condition_dim=4)
    model, uses_conditions = runner._point_model(bundle, "a6_point", None, None)
    assert uses_conditions and model.d_model == 32
    assert model.router.sender.num_heads == 2
    assert model.router.routers.shape[0] == 2


def test_development_rows_flag_boundary_censoring(tmp_path):
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    out = tmp_path / "runs"
    main(["--phase", "development", "--data_root", str(tmp_path),
          "--datasets", "DS03", "--methods", "a6_point",
          "--dev_path", str(dev), "--test_path", str(tmp_path / "absent.h5"),
          "--max_epochs", "2", "--patience", "5", "--folds", "1",
          "--output_dir", str(out), "--device", "cpu"])
    rows = list(csv.DictReader((out / "development.csv").open()))
    # With a 2-epoch cap every selected epoch lies within cap - 10.
    assert rows and all(row["boundary_censored"] == "True" for row in rows)
    assert all(row["stop_reason"] in {"cap", "patience"} for row in rows)
    assert all(int(row["stop_epoch"]) <= 2 for row in rows)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "partial_smoke"
    assert manifest["a6_point_dims"]["d_model"] == 32

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from test_data import _write_ncmapss_nine_units
from scripts.run_ncmapss_loso_final import main


def _selection(tmp_path, criterion="raw_interval_score"):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"status": "frozen", "datasets": {"DS03": {
        "search_units": [1, 2, 3, 5, 6, 8, 9], "calibration_units": [4, 7],
        "n_folds": 7, "criterion": criterion, "boundary_censored": False,
        "config": {"activation": "legacy", "lr": 0.001, "dropout": 0.1,
                   "weight_decay": 0.001},
        "budget_epoch": 2}}}))
    return path


def _common(tmp_path, selection, dev):
    return ["--dataset", "DS03", "--data_root", str(tmp_path),
            "--dev_path", str(dev), "--test_path", str(dev),
            "--selection_file", str(selection),
            "--output_dir", str(tmp_path / "final"), "--device", "cpu",
            "--touch_log", str(tmp_path / "touches.log")]


def test_a6_final_train_then_verified_one_shot_evaluation(tmp_path, capsys):
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    common = _common(tmp_path, _selection(tmp_path), dev)
    with pytest.raises(SystemExit):
        main(["--stage", "train", *common, "--eval_test"])
    main(["--stage", "train", *common])
    run = tmp_path / "final" / "DS03" / "seed_0"
    manifest = json.loads((run / "train_manifest.json").read_text())
    assert manifest["test_loaded"] is False and manifest["epochs_trained"] == 2
    assert manifest["train_units"] == [1, 2, 3, 5, 6, 8, 9]
    assert manifest["n_calibration_windows"] == 2 * 9          # G9 on engines 4 and 7
    assert not (tmp_path / "touches.log").exists()
    with pytest.raises(SystemExit):
        main(["--stage", "evaluate", *common])
    assert "--eval_test" in capsys.readouterr().err
    main(["--stage", "evaluate", *common, "--eval_test"])
    summary = json.loads((run / "evaluation" / "summary.json").read_text())
    assert summary["result_role"] == "descriptive_recheck"
    assert all(summary["verification"].values())
    assert summary["verification"]["calibration_reproduced"] is True
    rows = list(csv.DictReader((run / "evaluation" / "per_engine.csv").open()))
    assert rows and all(int(r["n_endpoints"]) == 4 for r in rows)
    assert "test access started" in (tmp_path / "touches.log").read_text()
    with pytest.raises(FileExistsError):
        main(["--stage", "evaluate", *common, "--eval_test"])   # one shot only


def test_a6_final_refuses_tampering_and_old_criterion(tmp_path):
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    common = _common(tmp_path, _selection(tmp_path), dev)
    main(["--stage", "train", *common])
    calibration = tmp_path / "final" / "DS03" / "seed_0" / "calibration_predictions.npz"
    calibration.write_bytes(calibration.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="calibration_hash"):
        main(["--stage", "evaluate", *common, "--eval_test"])
    assert not (tmp_path / "touches.log").exists()
    old = _common(tmp_path / "old", _selection(tmp_path, "pinball_mean"), dev)
    (tmp_path / "old").mkdir()
    with pytest.raises(ValueError, match="criterion"):
        main(["--stage", "train", *old])


def test_registered_calibration_units_read_from_p0_record(tmp_path, monkeypatch):
    from scripts import run_ncmapss_loso_final as final
    record = tmp_path / "results_v4/ncmapss_comparison/cqr/DS03/seed_0"
    record.mkdir(parents=True)
    (record / "cqr_variants_calibrators.json").write_text(json.dumps({
        "calibration_units": [4, 7], "val_units": [3, 5, 6]}))
    monkeypatch.setattr(final, "ROOT", tmp_path)
    assert final._registered_calibration_units("DS03") == [4, 7]


def test_a6_evaluate_rejects_changed_weights_before_test(tmp_path):
    import hashlib
    import torch
    dev = tmp_path / "N-CMAPSS_DS03-012.h5"
    _write_ncmapss_nine_units(dev)
    common = _common(tmp_path, _selection(tmp_path), dev)
    main(["--stage", "train", *common])
    run = tmp_path / "final" / "DS03" / "seed_0"
    state = torch.load(run / "best.pt", weights_only=False)
    for key, value in state["model_state"].items():
        if value.dtype.is_floating_point:
            state["model_state"][key] = value + 1.0        # a real change, not float noise
    torch.save(state, run / "best.pt")
    manifest = json.loads((run / "train_manifest.json").read_text())
    manifest["checkpoint_sha256"] = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    (run / "train_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="reproduces"):
        main(["--stage", "evaluate", *common, "--eval_test"])
    assert not (tmp_path / "touches.log").exists()           # aborted before test access

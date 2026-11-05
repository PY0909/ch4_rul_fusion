import json
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


def _tuning():
    path = Path(__file__).parents[1] / "scripts/run_ncmapss_a6_dev_tuning.py"
    assert path.exists(), "dev-only tuning entry point is missing"
    return importlib.import_module("scripts.run_ncmapss_a6_dev_tuning")


def test_commands_keep_the_registered_model_and_defer_test(tmp_path):
    tuning = _tuning()
    trials = tuning.commands("/data", python="python", output_root=tmp_path)
    assert len(trials) == 6
    expected = {
        "lr3e4": ("0.0003", "0.001", "0.1"),
        "wd3e3": ("0.001", "0.003", "0.1"),
        "dropout02": ("0.001", "0.001", "0.2"),
    }
    for dataset, name, command, run_dir in trials:
        assert dataset in {"DS03", "DS08a"}
        assert run_dir == tmp_path / name / dataset / "seed_0"
        assert command[command.index("--dataset") + 1] == dataset
        assert "--defer_test" in command
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--seed") + 1] == "0"
        assert [command[command.index(flag) + 1] for flag in
                ("--d_model", "--num_heads", "--num_routers", "--alpha")] == [
                    "32", "2", "2", "0.2"]
        assert tuple(command[command.index(flag) + 1] for flag in
                     ("--lr", "--weight_decay", "--dropout")) == expected[name]


def test_summary_keeps_missing_trials_and_existing_baseline(tmp_path):
    tuning = _tuning()
    baseline = tmp_path / "baseline"
    output = tmp_path / "tuning"
    source = baseline / "DS03/seed_0"
    source.mkdir(parents=True)
    (source / "calibration_only.json").write_text(json.dumps({
        "best_epoch": 26, "val_q50_RMSE": 17.87,
        "calibration_threshold": 16.63, "test_evaluated": False,
    }))
    rows = tuning.summary_rows(output, baseline)
    assert len(rows) == 8
    assert rows[0]["dataset"] == "DS03"
    assert rows[0]["config"] == "p0"
    assert rows[0]["status"] == "complete"
    assert rows[0]["best_epoch"] == 26
    assert rows[1]["status"] == "pending"
    assert rows[4]["dataset"] == "DS08a"
    assert rows[4]["status"] == "pending"


def test_dry_run_creates_no_output(tmp_path, capsys):
    tuning = _tuning()
    output = tmp_path / "tuning"
    tuning.main(["--data_root", "/data", "--output_root", str(output)])
    assert not output.exists()
    assert "--defer_test" in capsys.readouterr().out


def test_run_refuses_to_train_without_cuda(tmp_path, monkeypatch):
    tuning = _tuning()
    monkeypatch.setattr(tuning, "cuda_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA is unavailable"):
        tuning.main(["--data_root", "/data", "--output_root",
                     str(tmp_path / "tuning"), "--run"])

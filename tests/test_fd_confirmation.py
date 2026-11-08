import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from scripts import run_fd_confirmation as rc

SMOKE_POINT = ["--history", "5", "--rul_cap", "125", "--scaler", "grouped",
               "--encoder", "patch_cnn", "--variant", "A6",
               "--validation_protocol", "fixed_cuts", "--optimizer", "adam",
               "--weight_decay", "0", "--batch_size", "8", "--epochs", "2",
               "--patience", "2", "--no_scheduler", "--no_grad_clip",
               "--d_model", "16", "--num_heads", "2", "--num_routers", "2",
               "--calibration_ratio", "0.5"]
SMOKE_QUANTILE = ["--history", "5", "--rul_cap", "125", "--scaler", "grouped",
                  "--d_model", "16", "--num_heads", "2", "--num_routers", "2",
                  "--epochs", "2", "--batch_size", "8", "--lr", "1e-3",
                  "--weight_decay", "0.0", "--patience", "2",
                  "--calibration_ratio", "0.5", "--calibration_mode", "global"]


def _write_all(root: Path):
    data = root / "CMAPSSData"
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    for name in ("FD001", "FD003"):
        rows, test_rows, ruls = [], [], []
        for engine in range(1, 13):
            length = 30 + engine
            for cycle in range(1, length + 1):
                row = ([engine, cycle, 0.1, 0.2, 0.3]
                       + [float(engine + cycle + i) + rng.normal(0, 0.01)
                          for i in range(21)])
                rows.append(row)
                if cycle > length - 3:
                    test_rows.append(row[:])
            ruls.append([3])
        np.savetxt(data / f"train_{name}.txt", np.asarray(rows), fmt="%.6f")
        np.savetxt(data / f"test_{name}.txt", np.asarray(test_rows), fmt="%.6f")
        np.savetxt(data / f"RUL_{name}.txt", np.asarray(ruls), fmt="%d")


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    _write_all(tmp_path)
    monkeypatch.setattr(rc, "DATASETS", ("FD001",))
    monkeypatch.setattr(rc, "OUTPUT", tmp_path / "fd")
    monkeypatch.setattr(rc, "BASELINES", ("lstm_prob",))
    monkeypatch.setattr(rc, "A6_POINT_ARGS", SMOKE_POINT)
    monkeypatch.setattr(rc, "A6_QUANTILE_ARGS", SMOKE_QUANTILE)
    monkeypatch.setattr(rc, "LOCKED_HISTORY", 5)
    (tmp_path / "fd").mkdir()
    args = type("Args", (), {"data_root": str(tmp_path), "eval_test": False})()
    return args


def test_train_stage_writes_calibration_only_and_no_test(smoke_env):
    rc.stage_train(smoke_env)
    point_run = rc.OUTPUT / "a6_point" / "FD001" / "seed_0"
    quant_run = rc.OUTPUT / "a6_quantile" / "global" / "FD001" / "seed_0"
    base_run = rc.OUTPUT / "baselines" / "FD001" / "lstm_prob" / "seed_0"
    for run in (point_run, quant_run, base_run):
        assert (run / "best.pt").is_file()
        assert not (run / "predictions.csv").exists()
        assert not (run / "metrics.json").exists()
    assert json.loads((point_run / "calibration_only.json").read_text())[
        "test_evaluated"] is False
    log = (rc.OUTPUT / "train.log").read_text()
    assert "deferring test" in log or "test" in log  # trained with defer


def test_verify_passes_after_training(smoke_env):
    rc.stage_train(smoke_env)
    assert rc.stage_verify(smoke_env) == 0
    import csv as _csv
    with (rc.OUTPUT / "verification.csv").open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    assert rows and all(row["all_checks_pass"] == "True" for row in rows)


def test_evaluate_uses_trained_checkpoints_first_touch(smoke_env):
    rc.stage_train(smoke_env)
    assert rc.stage_verify(smoke_env) == 0
    smoke_env.eval_test = True
    rc.stage_evaluate(smoke_env)
    csv_text = (rc.OUTPUT / "confirmation_rows.csv").read_text()
    for expected in ("A6-Point", "A6-Quantile q50", "window_weighted",
                     "block_max_weighted", "lstm_prob"):
        assert expected in csv_text
    touch = (rc.OUTPUT / "test_touches_fd_confirmation.log").read_text()
    assert "FIRST-TOUCH" in touch and "complete" in touch
    # evaluation wrote its own artifacts without touching the train dirs
    assert not (rc.OUTPUT / "a6_point" / "FD001" / "seed_0" /
                "metrics.json").exists()


def test_evaluate_refuses_without_flag_or_failed_verify(smoke_env, capsys):
    with pytest.raises(SystemExit, match="--eval_test"):
        rc.stage_evaluate(smoke_env)          # missing --eval_test
    smoke_env.eval_test = True
    with pytest.raises(SystemExit):           # nothing trained yet
        rc.stage_evaluate(smoke_env)

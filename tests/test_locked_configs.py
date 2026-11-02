import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

LOCK_PATH = Path(__file__).parents[1] / "locked_configs.json"


def test_locked_configs_cover_all_datasets_with_protocol_flags():
    lock = json.loads(LOCK_PATH.read_text())
    assert lock["model"] == "A6"
    assert set(lock["datasets"]) == {"FD002", "FD004", "DS03", "DS08a"}
    for ds, entry in lock["datasets"].items():
        args = entry["args"]
        assert entry["params"] > 0 and entry["provenance"]
        assert "fixed_cuts" in entry["protocol"]
        assert "patch_cnn" in entry["protocol"] or (
            "--encoder" in args and "patch_cnn" in args)
        if ds.startswith("FD"):
            assert entry["runner"] == "rul_fusion.cli"
            assert "--validation_protocol" in args and "fixed_cuts" in args
            assert "--variant" in args and "A6" in args
            assert "--rul_cap" in args and "125" in args
            assert "--scaler" in args and "grouped" in args
            assert "--epochs" in args and "41" in args
        else:
            assert entry["runner"] == "scripts.run_ncmapss"
            # the N-CMAPSS runner hardcodes variant A6 and the fixed_cuts protocol
            assert "--model" in args and "A6" in args
            assert "--d_model" in args and "32" in args
            assert "--weight_decay" in args and "1e-3" in args
            # no RUL cap and no NASA in the N-CMAPSS runner
            assert "--rul_cap" not in args


def test_run_locked_dry_run_builds_expected_command():
    import subprocess
    for ds, marker in [("FD002", "--variant A6"), ("DS03", "--model A6")]:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).parents[1] / "scripts" / "run_locked.py"),
             "--dataset", ds, "--seed", "3", "--dry_run"],
            capture_output=True, text=True, check=True)
        out = proc.stdout
        assert f"--dataset {ds}" in out and "--seed 3" in out
        assert marker in out

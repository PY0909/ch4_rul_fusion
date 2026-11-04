import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts import run_ncmapss_comparison as comparison


def test_comparison_matrix_uses_registered_split_and_explicit_test():
    matrix = comparison.commands("/data", python="python")
    assert len(matrix) == 10
    point_commands = [c for c in matrix if c[1].endswith("run_ncmapss.py")]
    assert len(point_commands) == 4
    a6_commands = [c for c in point_commands if c[c.index("--model") + 1] == "A6"]
    for command in point_commands:
        assert "--registered_split" in command
        assert ["--d_model", "32", "--num_heads", "2", "--num_routers", "2"] == command[
            command.index("--d_model"):command.index("--d_model") + 6] if "--d_model" in command else True
    assert all(["--weight_decay", "1e-3"] == c[c.index("--weight_decay"):c.index("--weight_decay") + 2]
               for c in a6_commands)
    assert all("--defer_test" not in c for c in matrix)
    assert sum("--eval_test" in c for c in matrix) == 4
    assert not any("ensemble" in " ".join(c) for c in matrix)
    assert sum("run_ncmapss_cqr.py" in " ".join(c) for c in matrix) == 2
    assert all("--registered_split" not in c for c in matrix if "run_ncmapss_cqr.py" in " ".join(c))
    assert all("--registered_split" not in c for c in matrix if "run_ncmapss_probabilistic.py" in " ".join(c))


def test_summary_parses_metrics_and_cqr_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    root = tmp_path / "results_v4/ncmapss_comparison"
    run = root / "cqr/DS03/seed_0"
    run.mkdir(parents=True)
    (root / "hnn").mkdir()
    (root / "hnn/metrics.json").write_text(json.dumps({"dataset": "DS03", "MAE": 2.0}))
    (run / "cqr_variants_test_summary.json").write_text(json.dumps({
        "dataset": "DS03", "seed": 0, "official_test_evaluated": True,
        "q50_point": {"MAE": 1.5, "RMSE": 2.5},
        "rows": [{"variant": "global_block_max_cqr", "endpoint": "test", "PICP": 0.8}] }))
    rows = comparison._summary_rows(root)
    assert any(row.get("source") == "metrics.json" for row in rows)
    cqr = [row for row in rows if row.get("source") == "cqr_variants_test_summary.json"]
    assert len(cqr) == 1
    assert cqr[0]["PICP"] == 0.8 and cqr[0]["RMSE"] == 2.5

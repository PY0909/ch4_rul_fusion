import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from rul_fusion.cli import build_parser


def test_cli_help():
    root = Path(__file__).parents[1]
    proc = subprocess.run([sys.executable, "-m", "rul_fusion.cli", "--help"],
                          cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--dataset" in proc.stdout
    assert "--variant" in proc.stdout


def test_cli_accepts_ablation_variant_and_structural_controls():
    args = build_parser().parse_args([
        "--dataset", "FD002", "--data_root", "/tmp/data", "--variant", "A7",
        "--downsample_mode", "moment", "--graph_mode", "trend",
        "--readout", "single", "--condition_mode", "mean",
    ])
    assert args.variant == "A7"
    assert args.downsample_mode == "moment"
    assert args.graph_mode == "trend"
    assert args.readout == "single"
    assert args.condition_mode == "mean"

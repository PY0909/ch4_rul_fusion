from __future__ import annotations

import argparse
from pathlib import Path

from .baselines import DEFAULT_FC_STGNN_ROOT, FCSTGNNWrapper
from .data import build_cmapss_datasets, build_ncmapss_datasets
from .model import MultiScalePhaseGraphRUL
from .runner import fit_and_evaluate, set_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Point RUL prediction with multi-scale routing and sensor graph")
    parser.add_argument("--dataset", required=True, choices=["FD002", "FD004", "DS03", "DS08a"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model", default="fusion", choices=["fusion", "fc_stgnn"],
                        help="fusion: the multi-scale routed sensor-graph model; "
                             "fc_stgnn: unmodified FC-STGNN baseline trained by the same "
                             "runner (engine-grouped split, val-selected checkpoint, "
                             "single frozen test evaluation)")
    parser.add_argument("--fc_stgnn_root", default=str(DEFAULT_FC_STGNN_ROOT),
                        help="FC-STGNN repository directory containing Model.py")
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--rul_cap", type=int, default=0)
    parser.add_argument("--scaler", choices=["global", "grouped"], default="global",
                        help="global: z-score per column on train rows; "
                             "grouped: per-setting1-group min/max on sensors "
                             "(train engines only) + global min/max on conditions")
    parser.add_argument("--condition_features", choices=["settings", "settings_cycle"],
                        default="settings",
                        help="append the cycle column to the condition input "
                             "(ablation arm; FCSTGNN-style absolute-age cue)")
    parser.add_argument("--validation_protocol", choices=["legacy_endpoint", "fixed_cuts"],
                        default="legacy_endpoint",
                        help="legacy_endpoint: one uniformly random truncation per validation "
                             "engine; fixed_cuts: locked cuts at life fractions "
                             "{0.25, 0.5, 0.75, 0.9} per engine (registered A-series protocol)")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--calibration_ratio", type=float, default=None,
                        help="reserve part of the engine holdout for conformal calibration; "
                        "use the same value across chapter-four comparisons")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw",
                        help="adam + no scheduler/clip approximates the FCSTGNN recipe")
    parser.add_argument("--no_scheduler", action="store_true",
                        help="disable the ReduceLROnPlateau scheduler")
    parser.add_argument("--no_grad_clip", action="store_true",
                        help="disable gradient-norm clipping")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_routers", type=int, default=4)
    parser.add_argument("--graph_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha_shared", type=float, default=0.0)
    parser.add_argument("--alpha_specific", type=float, default=0.0)
    parser.add_argument("--encoder", choices=["linear", "patch_cnn"], default="linear",
                        help="linear: pointwise projection of (sensor, conditions); "
                             "patch_cnn: per-sensor temporal CNN stem (FC-STGNN-style, "
                             "BN over batch*sensors axis, conditions fused afterwards)")
    parser.add_argument("--input_norm", choices=["none", "bn"], default="none",
                        help="bn: extra BatchNorm1d over the sensor-token axis after "
                             "input encoding (isolates the FC-STGNN BatchNorm effect)")
    parser.add_argument("--variant", choices=[f"A{i}" for i in range(8)], default=None,
                        help="research preset A0-A7; when set, it selects readout, "
                             "downsampling, trend features and graph mode")
    parser.add_argument("--downsample_mode", choices=["legacy", "avg_endpoint", "learned", "moment"],
                        default="legacy", help="scale compression operator")
    parser.add_argument("--graph_mode", choices=["static", "level", "trend"], default="static",
                        help="static graph, level-dependent graph, or trend-reliability graph")
    parser.add_argument("--readout", choices=["legacy", "single"], default="legacy",
                        help="legacy duplicate summaries or one joint readout")
    parser.add_argument("--condition_mode", choices=["full", "mean", "none"], default="full",
                        help="condition sequence fusion mode")
    parser.add_argument("--trend_window_size", type=int, default=5,
                        help="causal filter width for learned/moment downsampling")
    parser.add_argument("--include_virtual_sensors", action="store_true")
    parser.add_argument("--disable_router", action="store_true")
    parser.add_argument("--disable_graph", action="store_true")
    parser.add_argument("--disable_shared_specific", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", default="results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.history is not None and args.history < 1:
        raise SystemExit("--history must be positive")
    if args.rul_cap < 0:
        raise SystemExit("--rul_cap must be non-negative")
    history = args.history or (50 if args.dataset.startswith("FD") else 16)
    if args.dataset.startswith("FD"):
        bundle = build_cmapss_datasets(args.data_root, args.dataset, history=history,
                                       rul_cap=args.rul_cap, split_seed=args.split_seed,
                                       calibration_ratio=args.calibration_ratio,
                                       scaler_mode=args.scaler,
                                       condition_features=args.condition_features,
                                       validation_protocol=args.validation_protocol)
    else:
        bundle = build_ncmapss_datasets(args.data_root, args.dataset, history=history,
                                         split_seed=args.split_seed,
                                         include_virtual=args.include_virtual_sensors,
                                         validation_protocol=args.validation_protocol)
    # The model contains randomly initialized modules; seed before construction
    # so variants are comparable and checkpoint initialization is reproducible.
    set_seed(args.seed)
    if args.model == "fc_stgnn":
        # The upstream model contains BatchNorm1d layers, so the last
        # incomplete training batch (possibly of size 1) is dropped.
        model = FCSTGNNWrapper(sensor_dim=bundle.sensor_dim, history=history,
                               dataset=args.dataset, repo_root=args.fc_stgnn_root)
        train_drop_last = True
    else:
        model = MultiScalePhaseGraphRUL(
            sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
            d_model=args.d_model, num_scales=args.num_scales, num_heads=args.num_heads,
            num_routers=args.num_routers, graph_layers=args.graph_layers,
            dropout=args.dropout, disable_router=args.disable_router,
            disable_graph=args.disable_graph, disable_shared_specific=args.disable_shared_specific,
            alpha_shared=args.alpha_shared, alpha_specific=args.alpha_specific,
            encoder=args.encoder, input_norm=args.input_norm, variant=args.variant,
            downsample_mode=args.downsample_mode, graph_mode=args.graph_mode,
            readout=args.readout, trend_window_size=args.trend_window_size,
            condition_mode=args.condition_mode,
        )
        train_drop_last = False
    run_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    config = vars(args)
    if args.model == "fc_stgnn":
        config = {**config, "fc_stgnn_hparams": model.hparams}
    fit_and_evaluate(bundle, model, run_dir, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
                     device=args.device, seed=args.seed, num_workers=args.num_workers,
                     train_drop_last=train_drop_last,
                     optimizer_name=args.optimizer, use_scheduler=not args.no_scheduler,
                     grad_clip=not args.no_grad_clip,
                     config=config)


if __name__ == "__main__":
    main()

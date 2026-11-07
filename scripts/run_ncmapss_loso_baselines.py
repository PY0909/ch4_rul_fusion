"""LOSO development and frozen final retraining for eight N-CMAPSS baselines.

Development never opens the test split. Final evaluation requires a frozen
selection file and --eval_test; scores from the previously observed test
endpoints are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
from datetime import datetime
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.metrics import regression_metrics, rmse  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.probabilistic_runner import fit_probabilistic  # noqa: E402
from rul_fusion.runner import set_seed  # noqa: E402
from scripts.run_ncmapss_probabilistic import NC_METHOD_DROPout, NC_METHOD_MC  # noqa: E402

SEARCH_UNITS = [1, 2, 3, 5, 6, 8, 9]
CALIBRATION_UNITS = [4, 7]
METHODS = ("a6_point", "fc_stgnn", "hnn", "edl_nig", "lstm_prob",
           "tcn_transformer_mc", "msauq", "bcpt")
POINT_METHODS = {"a6_point", "fc_stgnn"}
FC_DIMS = dict(indim_fea=2, Conv_out=4, lstmhidden_dim=8, lstmout_dim=12,
               conv_kernel=2, hidden_dim=8, time_length=8, num_node=14,
               num_windows=11, moving_window=[2, 2], stride=[1, 2],
               decay=0.7, pooling_choice="mean")
# P0 N-CMAPSS locks (results_v4/ncmapss_comparison/point/*/seed_0/config.json):
# A6-Point d32 / 2 heads / 2 routers with weight decay 1e-3; FC-STGNN wd 0.
A6_POINT_DIMS = dict(d_model=32, num_heads=2, num_routers=2)
POINT_WEIGHT_DECAY = {"a6_point": 1e-3, "fc_stgnn": 0.0}
# Common development budget shared with the A6-Quantile budget comparison
# (docs/ncmapss-loso-baseline-correction-amendment-2026-11-07.md).
DEV_MAX_EPOCHS = 120
DEV_PATIENCE = 30
BOUNDARY_MARGIN = 10
DEFAULT_FC_SOURCES = (
    Path("/home/v2/FCSTGNN/FC_STGNN"),
    Path("/Users/ppy/研/FCSTGNN-upstream/FC_STGNN"),
    Path("/Users/ppy/研/ch4-hcq-cqr-8382f94/FCSTGNN/FC_STGNN"),
)
SOURCE_FILES = (Path(__file__), ROOT / "rul_fusion/data.py",
                ROOT / "rul_fusion/model.py", ROOT / "rul_fusion/probabilistic.py",
                ROOT / "rul_fusion/probabilistic_runner.py",
                ROOT / "scripts/run_ncmapss_probabilistic.py")


class WindowToPatch(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, _w, y, unit, sid = self.base[index]
        history, sensors = x.shape
        if history != 16:
            raise ValueError("FC-STGNN's frozen NC dimensions require history 16")
        return x.reshape(8, 2, sensors).permute(0, 2, 1), y, unit, sid


def _fc_source(explicit):
    candidates = (Path(explicit),) if explicit else DEFAULT_FC_SOURCES
    for path in candidates:
        if (path / "Model.py").is_file() and (path / "Model_Base.py").is_file():
            return path.resolve()
    raise FileNotFoundError("FC-STGNN Model.py/Model_Base.py not found; pass --fc_source")


@contextmanager
def _fc_cpu_compat():
    """Keep upstream FC-STGNN's hard-coded CUDA allocations on CPU for smoke tests."""
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def _fc_model(source, sensor_dim, device):
    source = _fc_source(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))  # Model.py imports Model_Base by name.
    spec = importlib.util.spec_from_file_location("fcstgnn_loso_model", source / "Model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dims = {**FC_DIMS, "num_node": sensor_dim}
    with (_fc_cpu_compat() if device.type == "cpu" else nullcontext()):
        return module.FC_STGNN_RUL(n_class=1, **dims)


def _point_model(bundle, method, fc_source, device):
    if method == "fc_stgnn":
        return _fc_model(fc_source, bundle.sensor_dim, device), False
    return MultiScalePhaseGraphRUL(
        sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
        **A6_POINT_DIMS, dropout=0.1,
        encoder="patch_cnn", variant="A6", condition_mode="full"), True


@torch.no_grad()
def _point_predict(model, loader, device, conditions):
    model.eval()
    ys, ps, units, sids = [], [], [], []
    for batch in loader:
        if conditions:
            x, w, y, unit, sid = batch
            pred = model(x.to(device), w.to(device))
        else:
            x, y, unit, sid = batch
            with (_fc_cpu_compat() if device.type == "cpu" else nullcontext()):
                pred = model(x.to(device))
        ys.extend(y.numpy().tolist())
        ps.extend(pred.reshape(-1).cpu().numpy().tolist())
        units.extend(np.asarray(unit).reshape(-1).tolist())
        sids.extend(np.asarray(sid).reshape(-1).tolist())
    return tuple(np.asarray(values) for values in (ys, ps, units, sids))


def _train_point(bundle, method, output, *, epochs, patience, batch_size,
                 device, fc_source, use_validation):
    set_seed(0)
    model, conditions = _point_model(bundle, method, fc_source, device)
    model = model.to(device)
    data = bundle.train if conditions else WindowToPatch(bundle.train)
    train = DataLoader(data, batch_size=batch_size, shuffle=True,
                       drop_last=not conditions)
    if not len(train):
        raise ValueError("no training batches; reduce --batch_size")
    val = None
    if use_validation:
        val_data = bundle.val_endpoint if conditions else WindowToPatch(bundle.val_endpoint)
        val = DataLoader(val_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=POINT_WEIGHT_DECAY[method])
    loss_fn = nn.MSELoss()
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train:
            x = batch[0].to(device)
            y = (batch[2] if conditions else batch[1]).to(device)
            with (_fc_cpu_compat() if method == "fc_stgnn" and device.type == "cpu"
                  else nullcontext()):
                pred = model(x, batch[1].to(device)) if conditions else model(x)
            loss = loss_fn(pred.reshape(-1), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        row = {"epoch": epoch, "train_loss": total / count,
               "val_rmse": None}
        if val is not None:
            target, prediction, _, _ = _point_predict(model, val, device, conditions)
            row["val_rmse"] = rmse(target, prediction)
            if row["val_rmse"] < best:
                best, best_epoch, stale = row["val_rmse"], epoch, 0
                torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                            "selection": best, "selection_criterion": "val_endpoint_rmse"},
                           output / "best.pt")
            else:
                stale += 1
        history.append(row)
        if val is not None and stale >= patience:
            break
    if val is None:
        best_epoch = epochs
        torch.save({"model_state": model.state_dict(), "best_epoch": epochs,
                    "selection": None, "selection_criterion": "fixed_epoch"},
                   output / "best.pt")
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_rmse"))
        writer.writeheader()
        writer.writerows(history)
    result = {"best_epoch": best_epoch, "val_rmse": None if val is None else best,
              "selection_criterion": "fixed_epoch" if val is None else "val_endpoint_rmse"}
    return result


def _evaluate_point(bundle, method, checkpoint, output, *, batch_size, device,
                    fc_source):
    """Evaluate a verified fixed-epoch point checkpoint on the test endpoints once."""
    model, conditions = _point_model(bundle, method, fc_source, device)
    model = model.to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_data = (bundle.test_endpoint if conditions else
                 WindowToPatch(bundle.test_endpoint))
    target, prediction, units, sids = _point_predict(
        model, DataLoader(test_data, batch_size=batch_size), device, conditions)
    metrics = regression_metrics(target, prediction)
    metrics.pop("NASA_Score", None)
    result = {"best_epoch": int(state["best_epoch"]),
              "selection_criterion": state.get("selection_criterion", "fixed_epoch"),
              **metrics}
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("unit_id", "sample_id", "target_rul", "predicted_rul"))
        writer.writerows(zip(units, sids, target, prediction))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _prob_kwargs(method, args):
    return dict(seed=0, epochs=args.max_epochs, batch_size=args.batch_size,
                lr=1e-3, weight_decay=0.0 if method == "edl_nig" else 1e-3,
                patience=args.patience, dropout=NC_METHOD_DROPout[method],
                mc_samples=args.mc_samples or NC_METHOD_MC.get(method, 0),
                device=args.device)


def _identity(args, dataset, method, dev_path, phase, epochs, fc_source, selection):
    stat = dev_path.stat()
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in SOURCE_FILES}
    if method == "fc_stgnn":
        fc = _fc_source(fc_source)
        sources["FCSTGNN/Model.py"] = hashlib.sha256((fc / "Model.py").read_bytes()).hexdigest()
        sources["FCSTGNN/Model_Base.py"] = hashlib.sha256((fc / "Model_Base.py").read_bytes()).hexdigest()
    return {"dataset": dataset, "method": method, "phase": phase,
            "epochs": epochs, "patience": args.patience, "batch_size": args.batch_size,
            "history": 16, "mc_samples_override": args.mc_samples,
            "dev_path": str(dev_path), "dev_size": stat.st_size,
            "dev_mtime_ns": stat.st_mtime_ns, "source_hashes": sources,
            "selection_sha256": (hashlib.sha256(Path(selection).read_bytes()).hexdigest()
                                 if selection else None)}


def _bundle(args, dataset, partition, load_test):
    return build_ncmapss_datasets(
        args.data_root, dataset, history=16, split_seed=0,
        validation_protocol="fixed_cuts", unit_override=partition,
        dev_path=args.dev_path, test_path=args.test_path, load_test=load_test)


def _run_one(args, dataset, method, output, partition, epochs, phase, selection=None,
             checkpoint=None):
    """phase: development | final_train (never loads test) | final_eval."""
    dev_path = Path(args.dev_path or Path(args.data_root) / "data_set" /
                    ("N-CMAPSS_DS03-012.h5" if dataset == "DS03" else
                     "N-CMAPSS_DS08a-009.h5")).resolve()
    identity = _identity(args, dataset, method, dev_path, phase, epochs,
                         args.fc_source, selection)
    if checkpoint is not None:
        identity["evaluated_checkpoint_sha256"] = _sha256(checkpoint)
    required = {"development": ("best.pt", "history.csv"),
                "final_train": ("best.pt", "history.csv"),
                "final_eval": ("predictions.csv", "metrics.json")}[phase]
    if output.exists():
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"incomplete output directory: {output}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity or any(
                not (output / name).is_file() for name in required):
            raise RuntimeError(f"existing result has different identity: {output}")
        return manifest["result"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as tmp:
        staging = Path(tmp)
        bundle = _bundle(args, dataset, partition, phase == "final_eval")
        if phase != "final_eval" and getattr(bundle, "test_endpoint", None) is not None:
            raise RuntimeError("development and training stages must not load test")
        if bundle.metadata["dev_units"] != list(range(1, 10)):
            raise RuntimeError("N-CMAPSS baseline protocol requires dev units 1..9")
        if bundle.metadata["train_units"] != sorted(partition["train"]) or \
                bundle.metadata["val_units"] != sorted(partition["val"]):
            raise RuntimeError("dataset partition differs from registered units")
        device = torch.device(args.device)
        if method in POINT_METHODS:
            if phase == "final_eval":
                result = _evaluate_point(bundle, method, checkpoint, staging,
                                         batch_size=args.batch_size, device=device,
                                         fc_source=args.fc_source)
            else:
                result = _train_point(
                    bundle, method, staging, epochs=epochs, patience=args.patience,
                    batch_size=args.batch_size, device=device,
                    fc_source=args.fc_source, use_validation=phase == "development")
        else:
            result = fit_probabilistic(
                bundle, method, staging, **{**_prob_kwargs(method, args), "epochs": epochs},
                fixed_epochs=phase != "development",
                defer_test=phase != "final_eval",
                test_dataset=bundle.test_endpoint if phase == "final_eval" else None,
                evaluate_checkpoint=checkpoint if phase == "final_eval" else None)
            if phase == "development":
                selected = int(result["best_epochs"][0])
                with (staging / "history.csv").open(newline="") as handle:
                    history = list(csv.DictReader(handle))
                match = [row for row in history if int(row["epoch"]) == selected]
                if len(match) != 1:
                    raise RuntimeError("probabilistic best epoch is absent from history")
                result["best_epoch"] = selected
                result["val_rmse"] = float(match[0]["val_rmse"])
        final = phase != "development"
        manifest = {"identity": identity, "result": result, "phase": phase,
                    "train_units": sorted(partition["train"]),
                    "val_units": sorted(partition["val"]),
                    "calibration_units": sorted(partition["calibration"]) if final else [],
                    "calibration_units_excluded": CALIBRATION_UNITS,
                    "test_loaded": phase == "final_eval",
                    "test_evaluated": phase == "final_eval",
                    "test_result_role": ("descriptive_recheck" if phase == "final_eval"
                                         else None)}
        if phase == "final_train":
            manifest["checkpoint_sha256"] = _sha256(staging / "best.pt")
            manifest["epochs_trained"] = int(sum(1 for _ in csv.DictReader(
                (staging / "history.csv").open(newline=""))))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_final_train(train_dir, epochs, selection):
    """Checks that must pass before a checkpoint may be evaluated on test."""
    manifest = json.loads((train_dir / "manifest.json").read_text())
    checks = {
        "phase": manifest.get("phase") == "final_train",
        "test_not_loaded": manifest.get("test_loaded") is False,
        "frozen_epochs": (manifest["identity"].get("epochs") == epochs and
                          manifest.get("epochs_trained") == epochs),
        "selection_unchanged": (manifest["identity"].get("selection_sha256") ==
                                _sha256(selection)),
        "checkpoint_hash": manifest.get("checkpoint_sha256") == _sha256(train_dir / "best.pt"),
        "seven_search_engines": manifest.get("train_units") == SEARCH_UNITS,
        "no_validation_set": manifest.get("val_units") == [],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"{train_dir}: training verification failed: {failed}")
    return checks


def _read_selection(path, dataset, method):
    data = json.loads(Path(path).read_text())
    if data.get("status") != "frozen":
        raise ValueError("selection status must be frozen")
    entry = data.get("datasets", {}).get(dataset, {})
    if (entry.get("search_units") != SEARCH_UNITS or
            entry.get("calibration_units") != CALIBRATION_UNITS or
            entry.get("n_folds") != 7):
        raise ValueError("frozen selection lacks registered seven-fold protocol")
    epochs = entry.get("methods", {}).get(method, {}).get("epochs")
    if type(epochs) is not int or not 1 <= epochs <= DEV_MAX_EPOCHS:
        raise ValueError(f"frozen selection lacks valid epochs for {dataset}/{method}")
    return epochs


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=("development", "final_train", "final_eval"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=("DS03", "DS08a"),
                        default=["DS03", "DS08a"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dev_path")
    parser.add_argument("--test_path")
    parser.add_argument("--fc_source")
    parser.add_argument("--output_dir", default=str(
        ROOT / "results_v4/ncmapss_loso_baselines_v2"))
    parser.add_argument("--max_epochs", type=int, default=DEV_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEV_PATIENCE)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mc_samples", type=int, default=0,
                        help="synthetic smoke override; zero uses registered method setting")
    parser.add_argument("--folds", type=int, default=0,
                        help="first K folds for a partial smoke; zero runs all seven")
    parser.add_argument("--selection", help="frozen JSON required for final training")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--touch_log", default=str(
        ROOT / "results_v4/chapter4_main_logs/test_touches.log"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not 1 <= args.max_epochs <= DEV_MAX_EPOCHS or args.patience < 1 or args.batch_size < 1:
        parser.error(f"epochs, patience and batch size must be positive; "
                     f"epochs <= {DEV_MAX_EPOCHS}")
    if not 0 <= args.folds <= 7 or args.mc_samples < 0:
        parser.error("folds must be 0..7 and MC samples nonnegative")
    if args.phase == "final_train":
        if not args.selection or args.eval_test:
            parser.error("final_train requires --selection and never accesses test")
        if args.folds or args.mc_samples:
            parser.error("final stages do not allow fold or MC smoke overrides")
    elif args.phase == "final_eval":
        if not args.eval_test or not args.selection:
            parser.error("final_eval requires --selection and --eval_test")
        if args.folds or args.mc_samples:
            parser.error("final stages do not allow fold or MC smoke overrides")
    elif args.eval_test or args.selection:
        parser.error("development cannot load test or use a frozen selection")
    if args.phase == "development" and args.folds == 0 and \
            (args.max_epochs, args.patience, args.batch_size) != (
                DEV_MAX_EPOCHS, DEV_PATIENCE, 32):
        parser.error(f"complete development run requires {DEV_MAX_EPOCHS} epochs, "
                     f"patience {DEV_PATIENCE}, batch 32")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    final = args.phase in {"final_train", "final_eval"}
    frozen_epochs = ({(dataset, method): _read_selection(args.selection, dataset, method)
                      for dataset in args.datasets for method in args.methods}
                     if final else {})
    out = Path(args.output_dir)
    rows = []
    verification = {}
    if args.phase == "final_eval":
        # Verify every requested training run before touching test for any.
        for dataset in args.datasets:
            for method in args.methods:
                train_dir = out / "final" / dataset / method / "train"
                if not (train_dir / "manifest.json").is_file():
                    raise RuntimeError(f"missing verified training stage: {train_dir}")
                verification[(dataset, method)] = _verify_final_train(
                    train_dir, frozen_epochs[(dataset, method)], args.selection)
        touch = Path(args.touch_log)
        touch.parent.mkdir(parents=True, exist_ok=True)
        with touch.open("a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] "
                         f"ncmapss_loso_baselines final_eval: test access started for "
                         f"{','.join(args.datasets)} x {','.join(args.methods)} after "
                         f"verified fixed-epoch training (descriptive recheck)\n")
    for dataset in args.datasets:
        for method in args.methods:
            if args.phase == "development":
                held_units = SEARCH_UNITS[:args.folds] if args.folds else SEARCH_UNITS
                for held in held_units:
                    partition = {"train": [u for u in SEARCH_UNITS if u != held],
                                 "val": [held], "calibration": []}
                    run = out / "folds" / dataset / method / f"held_{held}"
                    result = _run_one(args, dataset, method, run, partition,
                                      args.max_epochs, "development")
                    epoch = result.get("best_epoch", result.get("best_epochs", [None])[0])
                    with (run / "history.csv").open(newline="") as handle:
                        stop_epoch = sum(1 for _ in csv.DictReader(handle))
                    rows.append({"dataset": dataset, "method": method,
                                 "held_unit": held, "best_epoch": epoch,
                                 "stop_epoch": stop_epoch,
                                 "stop_reason": ("cap" if stop_epoch >= args.max_epochs
                                                 else "patience"),
                                 "boundary_censored": (
                                     epoch is not None and
                                     int(epoch) > args.max_epochs - BOUNDARY_MARGIN),
                                 "selection_criterion": result["selection_criterion"],
                                 "val_rmse": result.get("val_rmse")})
                    print(f"[{dataset}/{method}] held={held} epoch={epoch}", flush=True)
            else:
                epochs = frozen_epochs[(dataset, method)]
                partition = {"train": SEARCH_UNITS, "val": [],
                             "calibration": CALIBRATION_UNITS}
                train_dir = out / "final" / dataset / method / "train"
                if args.phase == "final_train":
                    result = _run_one(args, dataset, method, train_dir, partition,
                                      epochs, "final_train", args.selection)
                    manifest = json.loads((train_dir / "manifest.json").read_text())
                    rows.append({"dataset": dataset, "method": method,
                                 "frozen_epochs": epochs,
                                 "epochs_trained": manifest["epochs_trained"],
                                 "checkpoint_sha256": manifest["checkpoint_sha256"],
                                 "test_loaded": manifest["test_loaded"]})
                    print(f"[{dataset}/{method}] trained {manifest['epochs_trained']}"
                          f"/{epochs} epochs; test not loaded", flush=True)
                else:
                    result = _run_one(args, dataset, method,
                                      out / "final" / dataset / method / "eval",
                                      partition, epochs, "final_eval", args.selection,
                                      checkpoint=train_dir / "best.pt")
                    rows.append({"dataset": dataset, "method": method,
                                 "epochs": epochs, "RMSE": result.get("RMSE"),
                                 "MAE": result.get("MAE"), "PICP": result.get("PICP"),
                                 "MPIW": result.get("MPIW"),
                                 "IntervalScore": result.get("IntervalScore"),
                                 "verification": ";".join(
                                     sorted(verification[(dataset, method)]))})
                    print(f"[{dataset}/{method}] fixed={epochs} RMSE={result.get('RMSE')}",
                          flush=True)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / {"development": "development.csv", "final_train": "final_train.csv",
                      "final_eval": "final_eval.csv"}[args.phase], rows)
    if args.phase == "development":
        summary = []
        for dataset in args.datasets:
            for method in args.methods:
                group = [row for row in rows if row["dataset"] == dataset and
                         row["method"] == method]
                summary.append({"dataset": dataset, "method": method,
                                "n_folds": len(group),
                                "candidate_fixed_epochs": (
                                    int(statistics.median_low(
                                        [int(row["best_epoch"]) for row in group]))
                                    if len(group) == 7 else None),
                                "mean_held_rmse": float(np.mean(
                                    [row["val_rmse"] for row in group])),
                                "n_boundary_censored": int(sum(
                                    bool(row["boundary_censored"]) for row in group)),
                                "n_stopped_by_cap": int(sum(
                                    row["stop_reason"] == "cap" for row in group)),
                                "selection_criterion": group[0]["selection_criterion"],
                                "selection_status": "development_candidate"})
        _write_csv(out / "development_summary.csv", summary)
        (out / "manifest.json").write_text(json.dumps({
            "phase": "development", "test_loaded": False,
            "amendment": "docs/ncmapss-loso-baseline-correction-amendment-2026-11-07.md",
            "search_units": SEARCH_UNITS,
            "calibration_units_excluded": CALIBRATION_UNITS,
            "status": (("complete" if set(args.methods) == set(METHODS)
                        else "complete_methods_subset")
                       if args.folds == 0 and len(args.datasets) == 2
                       else "partial_smoke"),
            "methods": list(args.methods),
            "max_epochs": args.max_epochs, "patience": args.patience,
            "boundary_margin": BOUNDARY_MARGIN,
            "a6_point_dims": A6_POINT_DIMS,
            "point_weight_decay": POINT_WEIGHT_DECAY,
            "method_specific_selection": True,
            "common_ranking_metric": "held_out_endpoint_rmse_after_frozen_selection",
        }, indent=2))


if __name__ == "__main__":
    main()

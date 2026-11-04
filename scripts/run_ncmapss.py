"""A6 (fusion variant) and FC-STGNN on N-CMAPSS DS03/DS08a, single seed.

Shared protocol: unit-cycle mean aggregation (X_s only, T excluded), global
z-score scaler fitted on train units, engine(unit)-grouped dev split
(split_seed=0), fixed_cuts endpoint validation ({0.25,0.5,0.75,0.9} per val
unit, raw flight RUL labels), checkpoint by endpoint RMSE, single frozen test
evaluation (one final window per official test unit). No RUL cap (the FD
cap=125 does not transfer; per the registered redesign plan).

Budget (registered for this N-CMAPSS screening, not transferred from FD):
Adam lr 1e-3, weight_decay 0, batch 32, 300 epochs, no scheduler/clip/early
stop (patience=300) — the dev split has ~7 train units (~500 unit-cycle
windows), so batch 100/41 epochs would give only ~200 optimizer steps.
FC-STGNN dims are adapted to history 16 (time_len 8 x patch 2), capacity kept
at the FD002 reference level; FC-STGNN consumes sensor windows only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/v2/FCSTGNN/FC_STGNN")
sys.path.insert(0, str(Path(__file__).parents[1]))

import Model as FCSTGNN  # noqa: E402
from rul_fusion.data import build_ncmapss_datasets  # noqa: E402
from rul_fusion.metrics import regression_metrics, rmse  # noqa: E402
from rul_fusion.model import MultiScalePhaseGraphRUL  # noqa: E402
from rul_fusion.runner import set_seed  # noqa: E402

# history 16 = time_len 8 x patch 2; MPNN outputs (8-2)/1+1=7 and (8-2)/2+1=4
FCSTGNN_DIMS = dict(indim_fea=2, Conv_out=4, lstmhidden_dim=8, lstmout_dim=12,
                    conv_kernel=2, hidden_dim=8, time_length=8, num_node=14,
                    num_windows=11, moving_window=[2, 2], stride=[1, 2],
                    decay=0.7, pooling_choice="mean")


class WindowToPatch(Dataset):
    """[history, sensors] -> [time_len, sensors, patch] for FC-STGNN."""

    def __init__(self, base, time_len: int):
        self.base = base
        self.time_len = time_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, _w, y, unit, sid = self.base[index]
        history, sensors = x.shape
        patch = history // self.time_len
        if patch * self.time_len != history:
            raise ValueError(f"history {history} not divisible by time_len {self.time_len}")
        return (x.reshape(self.time_len, patch, sensors).permute(0, 2, 1),
                y, unit, sid)


@torch.no_grad()
def evaluate(model, loader, device, takes_conditions: bool):
    model.eval()
    ys, ps, units, sids = [], [], [], []
    for batch in loader:
        if takes_conditions:
            x, w, y, unit, sid = batch
            pred = model(x.to(device), w.to(device))
        else:
            x, y, unit, sid = batch      # WindowToPatch yields (x, y, unit, sid)
            pred = model(x.to(device))
        pred = pred.squeeze(-1).cpu()
        ys.extend(y.numpy().tolist())
        ps.extend(pred.numpy().tolist())
        units.extend(np.asarray(unit).reshape(-1).tolist())
        sids.extend(np.asarray(sid).reshape(-1).tolist())
    return (np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64),
            np.asarray(units, dtype=np.int64), np.asarray(sids, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["DS03", "DS08a"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model", required=True, choices=["A6", "fc_stgnn"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_routers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--condition_mode", choices=["full", "mean", "none"], default="full")
    parser.add_argument("--registered_split", action="store_true",
                        help="use the registered 3-validation/2-calibration unit split")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    bundle = build_ncmapss_datasets(
        args.data_root, args.dataset, history=args.history, split_seed=0,
        val_ratio=(0.55 if args.registered_split else 0.2),
        calibration_ratio=(0.34 if args.registered_split else None),
        validation_protocol="fixed_cuts", registered_split=args.registered_split,
        min_calibration_units=(2 if args.registered_split else 1))
    train_loader = DataLoader(bundle.train, batch_size=args.batch_size, shuffle=True,
                              drop_last=(args.model == "fc_stgnn"))
    val_loader = DataLoader(bundle.val, batch_size=args.batch_size)
    endpoint_loader = DataLoader(bundle.val_endpoint, batch_size=args.batch_size)
    # Public-release test trajectories run to failure (final Y=0); under the
    # fixed_cuts protocol the frozen evaluation uses mid-life cuts instead.
    eval_test = (bundle.test_endpoint if getattr(bundle, "test_endpoint", None) is not None
                 else bundle.test)
    test_loader = DataLoader(eval_test, batch_size=args.batch_size)

    if args.model == "A6":
        model = MultiScalePhaseGraphRUL(
            sensor_dim=bundle.sensor_dim, condition_dim=bundle.condition_dim,
            d_model=args.d_model, num_heads=args.num_heads, num_routers=args.num_routers,
            dropout=args.dropout, encoder="patch_cnn", variant="A6",
            condition_mode=args.condition_mode).to(device)
        takes_conditions = True
    else:
        dims = dict(FCSTGNN_DIMS)
        dims["num_node"] = bundle.sensor_dim
        model = FCSTGNN.FC_STGNN_RUL(n_class=1, **dims).to(device)
        takes_conditions = False
        train_loader = DataLoader(WindowToPatch(bundle.train, 8),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(WindowToPatch(bundle.val, 8), batch_size=args.batch_size)
        endpoint_loader = DataLoader(WindowToPatch(bundle.val_endpoint, 8),
                                     batch_size=args.batch_size)
        test_loader = DataLoader(WindowToPatch(eval_test, 8), batch_size=args.batch_size)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    run_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            if takes_conditions:
                x, w, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                pred = model(x, w)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred.squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        train_loss = total / max(count, 1)
        val_y, val_p, _, _ = evaluate(model, val_loader, device, takes_conditions)
        val_rmse_all = rmse(val_y, val_p)
        ep_y, ep_p, _, _ = evaluate(model, endpoint_loader, device, takes_conditions)
        selection = rmse(ep_y, ep_p)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch:{epoch}, train_loss:{train_loss:.5f}, val_rmse_all:{val_rmse_all:.4f}, "
                  f"val_endpoint_rmse:{selection:.4f}, test_loss:deferred", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_rmse_all": val_rmse_all, "val_endpoint_rmse": selection})
        if selection < best:
            best, best_epoch = selection, epoch
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "selection": best}, run_dir / "best.pt")

    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_y, test_p, test_units, test_sids = evaluate(model, test_loader, device, takes_conditions)
    metrics = regression_metrics(test_y, test_p)
    # NASA score is a C-MAPSS convention; it is not reported for N-CMAPSS.
    metrics.pop("NASA_Score", None)
    result = {**metrics, "model": args.model, "dataset": args.dataset,
              "n_params": n_params, "best_epoch": int(best_epoch),
              "val_RMSE": float(best), "selection_criterion": "val_endpoint_rmse_fixed_cuts",
              "test_protocol": ("fixed_cuts mid-life endpoints" if getattr(bundle, "test_endpoint", None) is not None
                                else "final window"),
              "seed": args.seed, "device": str(device), "config": vars(args)}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with (run_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", "sample_id", "target_rul", "predicted_rul",
                         "absolute_error", "signed_error"])
        for unit, sid, y, p in zip(test_units, test_sids, test_y, test_p):
            writer.writerow([int(unit), int(sid), float(y), float(p),
                             abs(float(p - y)), float(p - y)])
    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rmse_all",
                                               "val_endpoint_rmse"])
        writer.writeheader()
        writer.writerows(history)
    (run_dir / "config.json").write_text(json.dumps({
        "model": args.model, "n_params": n_params,
        "fcstgnn_dims": FCSTGNN_DIMS if args.model == "fc_stgnn" else None,
        "args": vars(args),
        "protocol": "unit-cycle mean, X_s only, global z-score scaler on train units, "
                    "unit-grouped split (split_seed=0), fixed_cuts endpoint validation, "
                    "no RUL cap, single frozen test evaluation",
        "data": {k: v for k, v in bundle.metadata.items() if k not in {"val_endpoint", "test_endpoint"}},
        "endpoint_cuts": bundle.metadata["val_endpoint"],
    }, indent=2))
    print(f"MAE: {metrics['MAE']:.2f} RMSE: {metrics['RMSE']:.2f} "
          f"params: {n_params} best_epoch: {best_epoch}")


if __name__ == "__main__":
    main()

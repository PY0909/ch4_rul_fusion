"""FCSTGNN reference model under the rul_fusion frozen protocol.

Same data pipeline, split, labels, scaler and checkpoint selection as the
B0-B5 ablation runs (grouped min/max sensors, train-side rul_cap=125,
engine-grouped split, endpoint-val RMSE selection, test evaluated once on
raw RUL). FCSTGNN's own recipe (Adam, batch 100, per-dataset architecture
from main_RUL.py args_config) is kept for everything model-specific.
Note: FC_STGNN_RUL consumes sensor windows only; operating-condition inputs
are part of no arm here. Model_Base hard-codes .cuda(), so GPU is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/v2/FCSTGNN/FC_STGNN")
sys.path.insert(0, str(Path(__file__).parents[1]))

import Model as FCSTGNN  # noqa: E402
from rul_fusion.data import build_cmapss_datasets  # noqa: E402
from rul_fusion.metrics import regression_metrics, rmse  # noqa: E402
from rul_fusion.runner import set_seed  # noqa: E402

# main_RUL.py args_config: data_sub 2 (FD002) and 4 (FD004)
ARCH = {
    "FD002": dict(indim_fea=5, Conv_out=7, lstmhidden_dim=8, lstmout_dim=12,
                  conv_kernel=2, hidden_dim=8, time_length=10, num_node=14,
                  num_windows=14, moving_window=[2, 2], stride=[1, 2],
                  decay=0.7, pooling_choice="mean"),
    "FD004": dict(indim_fea=5, Conv_out=7, lstmhidden_dim=8, lstmout_dim=6,
                  conv_kernel=2, hidden_dim=8, time_length=10, num_node=14,
                  num_windows=14, moving_window=[2, 2], stride=[1, 2],
                  decay=0.7, pooling_choice="mean"),
}


class WindowToPatch(Dataset):
    """[history, sensors] -> [time_len, sensors, patch]; drops unused conditions."""

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
def evaluate(model, loader, device):
    model.eval()
    ys, ps, units, sids = [], [], [], []
    for x, y, unit, sid in loader:
        pred = model(x.to(device)).squeeze(-1).cpu()
        ys.extend(y.numpy().tolist())
        ps.extend(pred.numpy().tolist())
        units.extend(np.asarray(unit).reshape(-1).tolist())
        sids.extend(np.asarray(sid).reshape(-1).tolist())
    return (np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64),
            np.asarray(units, dtype=np.int64), np.asarray(sids, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["FD002", "FD004"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    bundle = build_cmapss_datasets(args.data_root, args.dataset, history=50,
                                   rul_cap=125, split_seed=0, scaler_mode="grouped")
    train_loader = DataLoader(WindowToPatch(bundle.train, 10), batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(WindowToPatch(bundle.val, 10), batch_size=args.batch_size)
    endpoint_loader = DataLoader(WindowToPatch(bundle.val_endpoint, 10),
                                 batch_size=args.batch_size)
    test_loader = DataLoader(WindowToPatch(bundle.test, 10), batch_size=args.batch_size)

    model = FCSTGNN.FC_STGNN_RUL(n_class=1, **ARCH[args.dataset]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    run_dir = Path(args.output_dir) / args.dataset / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        skipped = 0
        for x, y, _, _ in train_loader:
            if x.size(0) < 2:  # BatchNorm1d in nonlin_map2 needs >1 sample
                skipped += x.size(0)
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        train_loss = total / max(count, 1)
        val_y, val_p, _, _ = evaluate(model, val_loader, device)
        val_rmse_all = rmse(val_y, val_p)
        ep_y, ep_p, _, _ = evaluate(model, endpoint_loader, device)
        selection = rmse(ep_y, ep_p)
        print(f"epoch:{epoch}, train_loss:{train_loss:.5f}, val_rmse_all:{val_rmse_all:.4f}, "
              f"val_endpoint_rmse:{selection:.4f}, test_loss:deferred"
              + (f" (skipped {skipped} singleton batches)" if skipped else ""))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_rmse_all": val_rmse_all, "val_endpoint_rmse": selection})
        if selection < best:
            best, best_epoch, stale = selection, epoch, 0
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "selection": best}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping with best_ls:{best:.5f} (val_endpoint_rmse)")
                break

    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_y, test_p, test_units, test_sids = evaluate(model, test_loader, device)
    metrics = regression_metrics(test_y, test_p)
    result = {**metrics, "model": "FC_STGNN_RUL", "dataset": args.dataset,
              "n_params": n_params, "best_epoch": int(best_epoch),
              "val_RMSE": float(best), "selection_criterion": "val_endpoint_rmse",
              "seed": args.seed, "device": str(device), "skipped_singleton_batches": True}
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
        "model": "FC_STGNN_RUL", "arch": ARCH[args.dataset], "n_params": n_params,
        "optimizer": "adam", "lr": args.lr, "batch_size": args.batch_size,
        "epochs": args.epochs, "patience": args.patience,
        "protocol": "grouped min/max scaler, rul_cap=125 train side, engine-grouped split "
                    "(split_seed=0), endpoint-val selection, test raw RUL evaluated once",
        "data": {k: v for k, v in bundle.metadata.items() if k != "val_endpoint"},
        "endpoint_cuts": bundle.metadata["val_endpoint"],
    }, indent=2))
    print(f"MAE: {metrics['MAE']:.2f} RMSE: {metrics['RMSE']:.2f} "
          f"NASA_Score: {metrics['NASA_Score']:.2f} params: {n_params}")


if __name__ == "__main__":
    main()

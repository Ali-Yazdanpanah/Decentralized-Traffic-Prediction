"""
Federated learning (FedAvg) over fog zones, matching Phase 1 train/test split.
Local objective: MSE on the |zone| x |zone| submatrix; global eval: full-matrix MSE.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def _load_baseline_module() -> ModuleType:
    mod_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "2_train_baseline.py"
    )
    name = "train_baseline"
    spec = importlib.util.spec_from_file_location(name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_baseline = _load_baseline_module()
CentralizedSTGNN = _baseline.CentralizedSTGNN
TrafficWindowDataset = _baseline.TrafficWindowDataset
_adj_to_edge_index = _baseline._adj_to_edge_index
_minmax = _baseline._minmax
_fit_minmax = _baseline._fit_minmax

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
SAVED = PROJECT_ROOT / "saved_models"
LOGS = PROJECT_ROOT / "logs"

# Match baseline hyperparameters
WINDOW = _baseline.WINDOW
HIDDEN_DIM = _baseline.HIDDEN_DIM
BATCH_SIZE = _baseline.BATCH_SIZE
LR = _baseline.LR
COMMUNICATION_ROUNDS = 20
LOCAL_EPOCHS = 3
N_ZONES = 4


def _load_node_indices_for_zones() -> list[torch.Tensor]:
    with open(PROCESSED / "fog_topology.json", encoding="utf-8") as f:
        fog: dict = json.load(f)
    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        name_to_idx: dict = json.load(f)
    out: list[torch.Tensor] = []
    for z in range(N_ZONES):
        key = f"zone_{z}"
        if key not in fog:
            raise KeyError(f"Missing {key} in fog_topology.json")
        names: list = fog[key]["nodes"]
        idx = sorted(int(name_to_idx[n]) for n in names)
        out.append(torch.tensor(idx, dtype=torch.long))
    return out


class LocalClient:
    def __init__(
        self,
        zone_id: int,
        zone_idx: torch.Tensor,
        model: CentralizedSTGNN,
    ) -> None:
        self.zone_id = zone_id
        self.zone_idx = zone_idx
        self.model = model

    @torch.inference_mode()
    def set_weights(self, state: dict) -> None:
        self.model.load_state_dict(state)

    def get_state(self) -> dict:
        return self.model.state_dict()

    def train_one_round(
        self,
        train_loader: DataLoader,
        edge_index: torch.Tensor,
        device: torch.device,
        loss_module: nn.MSELoss,
    ) -> None:
        self.model.train()
        z = self.zone_idx.to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        mse = loss_module
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = self.model(xb, edge_index)
                pz = pred.index_select(1, z).index_select(2, z)
                yz = yb.index_select(1, z).index_select(2, z)
                loss = mse(pz, yz)
                loss.backward()
                opt.step()


def _fedavg(state_dicts: List[dict], n: int) -> dict:
    keys = state_dicts[0].keys()
    out: dict = {}
    for k in keys:
        stack = [sd[k] for sd in state_dicts]
        out[k] = sum(stack) / float(n)
    return out


@torch.inference_mode()
def _evaluate_full(
    model: CentralizedSTGNN,
    data_loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    loss_fn: nn.MSELoss,
) -> float:
    model.eval()
    tot = 0.0
    count = 0.0
    mse = loss_fn
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb, edge_index)
        loss = mse(pred, yb)
        tot += loss.item() * xb.size(0)
        count += xb.size(0)
    return tot / max(count, 1.0)


@torch.inference_mode()
def _load_baseline_test_loss(
    n: int,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    loss_fn: nn.MSELoss,
) -> float | None:
    p = SAVED / "centralized_baseline.pth"
    if not p.is_file():
        return None
    m = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
    try:
        ck = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(p, map_location=device)
    m.load_state_dict(ck["model_state_dict"])
    return _evaluate_full(m, test_loader, edge_index, device, loss_fn)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    SAVED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    traffic = np.load(PROCESSED / "traffic_tensor.npy")
    n = int(adj.shape[0])
    t_total = int(traffic.shape[0])
    w = WINDOW
    n_samples = t_total - w
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    print(
        f"Window={w} | Samples: {n_samples} (train {n_train}, test {n_test} = {100.0 * n_test / n_samples:.1f}%)",
        flush=True,
    )

    mm_path = PROCESSED / "traffic_minmax.json"
    if mm_path.is_file():
        with open(mm_path, encoding="utf-8") as f:
            j = json.load(f)
        t_min, t_max = float(j["min"]), float(j["max"])
    else:
        train_X_stack = [traffic[i : i + w] for i in range(n_train)]
        train_y_stack = [traffic[i + w] for i in range(n_train)]
        arr_x = np.stack(train_X_stack, axis=0)
        arr_y = np.stack(train_y_stack, axis=0)
        t_min, t_max = _fit_minmax(
            [arr_x.reshape(-1), arr_y.reshape(-1)]
        )

    traffic_n = _minmax(traffic, t_min, t_max).astype(np.float32)
    full_ds = TrafficWindowDataset(traffic_n, w)
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_samples))
    train_ds = Subset(full_ds, train_indices)
    test_ds = Subset(full_ds, test_indices)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    edge_index = _adj_to_edge_index(adj).to(device)
    mse = nn.MSELoss()

    zone_index_tensors = _load_node_indices_for_zones()
    global_model = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
    clients: list[LocalClient] = []
    for z in range(N_ZONES):
        c_model = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
        c_model.load_state_dict(global_model.state_dict())
        clients.append(
            LocalClient(
                z,
                zone_index_tensors[z].clone().cpu(),
                c_model,
            )
        )

    progress: list[tuple[int, float]] = []
    for r in range(1, COMMUNICATION_ROUNDS + 1):
        gstate = global_model.state_dict()
        for c in clients:
            c.set_weights(gstate)
        for c in clients:
            c.train_one_round(train_loader, edge_index, device, mse)
        merged = _fedavg([c.get_state() for c in clients], N_ZONES)
        global_model.load_state_dict(merged)
        test_loss = _evaluate_full(
            global_model, test_loader, edge_index, device, mse
        )
        progress.append((r, test_loss))
        print(
            f"Round {r:2d} / {COMMUNICATION_ROUNDS} | Global test loss: {test_loss:.6f}",
            flush=True,
        )

    out_model = SAVED / "federated_final.pth"
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "n": n,
            "hidden_dim": HIDDEN_DIM,
            "window": w,
            "communication_rounds": COMMUNICATION_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
        },
        out_model,
    )
    csv_path = LOGS / "federated_progress.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["Round", "Test_Loss"])
        for rnd, tloss in progress:
            wri.writerow([rnd, f"{tloss:.8f}"])

    baseline_loss = _load_baseline_test_loss(
        n, test_loader, edge_index, device, mse
    )
    final_loss = progress[-1][1] if progress else float("nan")
    b_str = f"{baseline_loss:.6f}" if baseline_loss is not None else "N/A (run Phase 1 baseline or ensure saved_models/centralized_baseline.pth exists)"
    print(
        f"Centralized Baseline Loss: {b_str} | Federated Final Loss: {final_loss:.6f}",
        flush=True,
    )
    print(f"Saved: {out_model}\nSaved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Centralized spatiotemporal GNN baseline: GCN (spatial) + GRU (temporal).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.nn import GCNConv
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install torch_geometric: pip install torch_geometric") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
SAVED = PROJECT_ROOT / "saved_models"
LOGS = PROJECT_ROOT / "logs"
WINDOW = 6
HIDDEN_DIM = 64
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20


def _adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Undirected 0/1 adjacency -> COO [2, E] with both directions, plus self-loops."""
    n = adj.shape[0]
    rows, cols = np.where(adj == 1)
    ei = np.stack([rows, cols], axis=0).astype(np.int64)
    # self-loops for GCN
    sl = np.arange(n, dtype=np.int64)
    self_ei = np.stack([sl, sl], axis=0)
    ei = np.concatenate([ei, self_ei], axis=1)
    return torch.from_numpy(ei).long()


def _batch_edge_index(edge_index: torch.Tensor, n: int, batch_size: int) -> torch.Tensor:
    """Repeat edge_index for each graph in the batch with node offset."""
    if batch_size == 1:
        return edge_index
    parts = [edge_index + b * n for b in range(batch_size)]
    return torch.cat(parts, dim=1)


class TrafficWindowDataset(Dataset):
    """Sliding window: X (W, N, N) -> Y (N, N) next step."""

    def __init__(self, traffic: np.ndarray, w: int) -> None:
        """
        traffic: (T, N, N) already normalized
        """
        self.w = w
        self.t = traffic
        self.t_steps = traffic.shape[0] - w
        if self.t_steps <= 0:
            raise ValueError("Not enough time steps for the window size")

    def __len__(self) -> int:
        return self.t_steps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.t[idx : idx + self.w]  # (W, N, N)
        y = self.t[idx + self.w]  # (N, N)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class CentralizedSTGNN(nn.Module):
    def __init__(self, n_nodes: int, hidden_dim: int) -> None:
        super().__init__()
        self.n = n_nodes
        self.hidden_dim = hidden_dim
        self.gcn = GCNConv(n_nodes, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, n_nodes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, W, N, N) — per time step, node i features = row i (traffic to all N nodes)
        edge_index: (2, E) for a single graph (no batch offset)
        """
        b, w, n, _ = x.shape
        h_steps: list[torch.Tensor] = []
        ei = _batch_edge_index(edge_index, n, b)
        for t in range(w):
            x_t = x[:, t, :, :]  # (B, N, N)
            x_flat = x_t.reshape(b * n, n)
            h = self.gcn(x_flat, ei)
            h = F.relu(h)
            h = h.view(b, n, self.hidden_dim)
            h_steps.append(h)
        # (B, N, W, H)
        h_seq = torch.stack(h_steps, dim=2)
        # (B * N, W, H) for GRU over time per node
        gru_in = h_seq.permute(0, 1, 2, 3).contiguous().view(b * n, w, self.hidden_dim)
        out, _ = self.gru(gru_in)
        last = out[:, -1, :]  # (B * N, H)
        pred = self.out(last)  # (B * N, N)
        return pred.view(b, n, n)


def _fit_minmax(
    parts: list[np.ndarray],
) -> tuple[float, float]:
    lo = min(float(p.min()) for p in parts)
    hi = max(float(p.max()) for p in parts)
    if hi - lo < 1e-12:
        return lo, hi + 1e-6
    return lo, hi


def _minmax(
    a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (a - lo) / (hi - lo)


def main() -> None:
    history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    traffic = np.load(PROCESSED / "traffic_tensor.npy")  # (T, N, N)
    n = int(adj.shape[0])
    t_total = int(traffic.shape[0])
    w = WINDOW

    edge_index = _adj_to_edge_index(adj).to(device)
    n_samples = t_total - w
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    print(f"Samples: {n_samples} (train {n_train}, test {n_test})")

    train_X_stack = [
        traffic[i : i + w] for i in range(n_train)
    ]
    train_y_stack = [
        traffic[i + w] for i in range(n_train)
    ]
    arr_x = np.stack(train_X_stack, axis=0)
    arr_y = np.stack(train_y_stack, axis=0)
    t_min, t_max = _fit_minmax(
        [arr_x.reshape(-1), arr_y.reshape(-1)]
    )
    with open(PROCESSED / "traffic_minmax.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "min": t_min,
                "max": t_max,
                "fit_on": "train_split_windows",
                "window": w,
            },
            f,
            indent=2,
        )

    traffic_n = _minmax(traffic, t_min, t_max).astype(np.float32)
    full_ds = TrafficWindowDataset(traffic_n, w)
    train_ds = torch.utils.data.Subset(full_ds, range(n_train))
    test_ds = torch.utils.data.Subset(full_ds, range(n_train, n_samples))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    model = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        n_b = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb, edge_index)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n_b += xb.size(0)
        tr_loss /= max(n_b, 1)

        model.eval()
        te_loss = 0.0
        n_t = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb, edge_index)
                loss = loss_fn(pred, yb)
                te_loss += loss.item() * xb.size(0)
                n_t += xb.size(0)
        te_loss /= max(n_t, 1)
        print(
            f"Epoch {epoch + 1:2d} | Train Loss: {tr_loss:.6f} | Test Loss: {te_loss:.6f}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "test_loss": te_loss,
            }
        )

    LOGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(LOGS / "centralized_baseline.csv", index=False)

    SAVED.mkdir(parents=True, exist_ok=True)
    out_path = SAVED / "centralized_baseline.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n": n,
            "hidden_dim": HIDDEN_DIM,
            "window": w,
        },
        out_path,
    )
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()

"""
Executive summary plots for final research reporting.

Tasks:
1) Pareto frontier: bandwidth savings (%) vs final MSE.
2) Spatial node-wise error distribution on federated test predictions.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def _load_baseline_module() -> ModuleType:
    p = Path(__file__).resolve().parent / "models" / "2_train_baseline.py"
    spec = importlib.util.spec_from_file_location("train_baseline", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b = _load_baseline_module()
CentralizedSTGNN = _b.CentralizedSTGNN
TrafficWindowDataset = _b.TrafficWindowDataset
_adj_to_edge_index = _b._adj_to_edge_index
_minmax = _b._minmax
WINDOW = _b.WINDOW
HIDDEN_DIM = _b.HIDDEN_DIM
BATCH_SIZE = _b.BATCH_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS = PROJECT_ROOT / "logs"
FIGURES = PROJECT_ROOT / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SAVED = PROJECT_ROOT / "saved_models"


def _set_plot_style() -> None:
    for st in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        if st in plt.style.available or st == "default":
            plt.style.use(st)
            break


def task1_pareto_frontier() -> None:
    p = LOGS / "systems_ablation_summary.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    x = df["Estimated_Bandwidth_Savings_pct"].values.astype(float)
    y = df["Final_MSE"].values.astype(float)
    labels = df["Strategy"].values.tolist()

    short = []
    for s in labels:
        if s.lower().startswith("high"):
            short.append("High")
        elif s.lower().startswith("medium"):
            short.append("Medium")
        elif s.lower().startswith("low"):
            short.append("Low")
        else:
            short.append(s)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=160)
    ax.scatter(x, y, s=120, c=["C3", "C0", "C2"], alpha=0.95)
    for xi, yi, txt in zip(x, y, short):
        ax.annotate(
            txt,
            (xi, yi),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
            weight="bold",
        )
    ax.set_xlabel("Estimated bandwidth savings (%)")
    ax.set_ylabel("Final test MSE")
    ax.set_title("Pareto frontier: communication efficiency vs prediction error")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = FIGURES / "pareto_efficiency.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _load_test_data() -> tuple[DataLoader, torch.Tensor, dict[int, str], dict[str, int], dict[str, int]]:
    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    traffic = np.load(PROCESSED / "traffic_tensor.npy")
    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        name_to_idx: dict[str, int] = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    with open(PROCESSED / "fog_topology.json", encoding="utf-8") as f:
        fog = json.load(f)
    zone_of: dict[str, int] = {}
    for z in range(4):
        for node in fog[f"zone_{z}"]["nodes"]:
            zone_of[node] = z

    mm_path = PROCESSED / "traffic_minmax.json"
    if not mm_path.is_file():
        raise FileNotFoundError("Missing traffic_minmax.json")
    with open(mm_path, encoding="utf-8") as f:
        j = json.load(f)
    t_min, t_max = float(j["min"]), float(j["max"])
    traffic_n = _minmax(traffic, t_min, t_max).astype(np.float32)

    n = int(adj.shape[0])
    n_samples = int(traffic_n.shape[0] - WINDOW)
    n_train = int(0.8 * n_samples)

    full_ds = TrafficWindowDataset(traffic_n, WINDOW)
    test_ds = Subset(full_ds, list(range(n_train, n_samples)))
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    edge_index = _adj_to_edge_index(adj)
    return test_loader, edge_index, idx_to_name, name_to_idx, zone_of


@torch.inference_mode()
def task2_spatial_error_distribution() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = SAVED / "federated_final.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing {ckpt}; run federated training first.")

    test_loader, edge_index, idx_to_name, _name_to_idx, zone_of = _load_test_data()
    edge_index = edge_index.to(device)
    n_nodes = len(idx_to_name)
    model = CentralizedSTGNN(n_nodes, HIDDEN_DIM).to(device)
    try:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Node-wise MSE over test set:
    # MSE_i = mean_{batch,tgt} (pred[:, i, :] - y[:, i, :])^2
    sse = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    count = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb, edge_index)
        diff2 = (pred - yb) ** 2  # (B, N, N)
        # sum over batch and target-dim
        sse += diff2.sum(dim=(0, 2)).float()
        count += float(diff2.shape[0] * diff2.shape[2])
    node_mse = (sse / torch.clamp(count, min=1.0)).cpu().numpy()

    names = [idx_to_name[i] for i in range(n_nodes)]
    zones = [zone_of.get(nm, -1) for nm in names]
    cmap = plt.cm.tab10
    colors = [cmap(z % 10) if z >= 0 else (0.5, 0.5, 0.5, 1.0) for z in zones]

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    x = np.arange(n_nodes)
    ax.bar(x, node_mse, color=colors, edgecolor="0.25", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Node-wise test MSE")
    ax.set_xlabel("Node ID")
    ax.set_title("Spatial error distribution (colored by Fluid zone)")
    ax.grid(True, axis="y", alpha=0.3)
    # Legend proxies for zone colors
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=cmap(z), edgecolor="none", label=f"Zone {z}") for z in sorted(set([z for z in zones if z >= 0]))]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out = FIGURES / "spatial_error_distribution.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    task1_pareto_frontier()
    task2_spatial_error_distribution()


if __name__ == "__main__":
    main()

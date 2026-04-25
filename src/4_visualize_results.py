"""
Load experiment logs and produce figures for README (see `README.md`).
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

def _load_baseline_module() -> ModuleType:
    p = (
        Path(__file__).resolve().parent
        / "models"
        / "2_train_baseline.py"
    )
    name = "train_baseline"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_baseline = _load_baseline_module()
CentralizedSTGNN = _baseline.CentralizedSTGNN
TrafficWindowDataset = _baseline.TrafficWindowDataset
_adj_to_edge_index = _baseline._adj_to_edge_index
_minmax = _baseline._minmax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SAVED = PROJECT_ROOT / "saved_models"

HIDDEN_DIM = _baseline.HIDDEN_DIM
WINDOW = _baseline.WINDOW


def task1_convergence() -> None:
    c = pd.read_csv(LOGS / "centralized_baseline.csv")
    f = pd.read_csv(LOGS / "federated_progress.csv")
    t_col = "Test_Loss" if "Test_Loss" in f.columns else f.columns[1]
    c_loss = c["test_loss"].values
    f_loss = f[t_col].values
    n = min(len(c_loss), len(f_loss))
    progress = np.arange(1, n + 1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for st in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        if st in plt.style.available or st == "default":
            plt.style.use(st)
            break
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        progress,
        c_loss[:n],
        "r--",
        linewidth=2.0,
        label="Centralized Baseline (Target)",
        alpha=0.9,
    )
    ax.plot(
        progress,
        f_loss[:n],
        "b-",
        linewidth=2.0,
        label="Federated progress (FedAvg)",
        alpha=0.9,
    )
    ax.set_xlabel("Training progress", fontsize=11)
    ax.set_ylabel("Test MSE (full matrix)", fontsize=11)
    ax.set_title("Convergence: central vs. federated test error", fontsize=12)
    if n > 0:
        step = max(1, (n + 1) // 8)
        ticks = list(range(1, n + 1, step))
        if ticks[-1] != n:
            ticks.append(n)
        ax.set_xticks(ticks)
    ax.set_xlim(0.5, n + 0.5)
    ax.legend(frameon=True, loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = FIGURES / "convergence_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def _prepare_test_data() -> tuple:
    """Returns traffic_n, full_ds, n_train, n_samples, n, w, test_subset_indices."""
    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    traffic = np.load(PROCESSED / "traffic_tensor.npy")
    n = int(adj.shape[0])
    t_total = int(traffic.shape[0])
    w = WINDOW
    n_samples = t_total - w
    n_train = int(0.8 * n_samples)
    mm_path = PROCESSED / "traffic_minmax.json"
    with open(mm_path, encoding="utf-8") as f:
        j = json.load(f)
    t_min, t_max = float(j["min"]), float(j["max"])
    traffic_n = _minmax(traffic, t_min, t_max).astype(np.float32)
    full_ds = TrafficWindowDataset(traffic_n, w)
    return traffic_n, full_ds, n_train, n_samples, n, w


@torch.inference_mode()
def task2_traffic_sample() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        name_to_idx: dict = json.load(f)
    if "de1.de" not in name_to_idx:
        raise KeyError("de1.de not in node_mapping.json")
    i_de = int(name_to_idx["de1.de"])

    _traffic_n, full_ds, n_train, n_samples, n, w = _prepare_test_data()
    n_test = n_samples - n_train
    if n_test < 100:
        raise ValueError("Need at least 100 test windows")

    p_path = SAVED / "federated_final.pth"
    if not p_path.is_file():
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        ax.text(
            0.5,
            0.5,
            "federated_final.pth not found\nRun: python src/federated/2_train_federated.py",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        out = FIGURES / "traffic_prediction_sample.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(
            f"Wrote placeholder to {out} (train federated to replace).",
            flush=True,
        )
        return

    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    edge_index = _adj_to_edge_index(adj).to(device)
    model = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
    try:
        ck = torch.load(p_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(p_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Last 100 test windows: dataset indices in full_ds are n_test_100 to end
    start = n_train + n_test - 100
    indices = list(range(start, n_train + n_test))
    gt: list[float] = []
    pr: list[float] = []
    for idx in indices:
        x, y = full_ds[idx]
        xb = x.unsqueeze(0).to(device)
        y = y.to(device)
        pred = model(xb, edge_index)
        p0 = pred[0]
        # total outgoing (normalized) from de1.de: row sum
        gt.append(float(y[i_de, :].sum().cpu().item()))
        pr.append(float(p0[i_de, :].sum().cpu().item()))

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    t_axis = np.arange(100)
    ax.plot(t_axis, gt, color="0.2", linewidth=1.2, label="Ground truth (row sum, normalized)")
    ax.plot(
        t_axis,
        pr,
        color="C0",
        linewidth=1.2,
        linestyle="-",
        label="Federated prediction (row sum, normalized)",
    )
    ax.set_xlabel("Time index (last 100 test windows, ~15 min / step, ≈ 25 h total)", fontsize=10)
    ax.set_ylabel("∑_j T(de1.de → j) (min–max scaled)", fontsize=10)
    ax.set_title("Traffic at aggregator node de1.de — last 100 test windows", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "traffic_prediction_sample.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


def _final_losses() -> tuple[float, float, float]:
    c = pd.read_csv(LOGS / "centralized_baseline.csv")
    f = pd.read_csv(LOGS / "federated_progress.csv")
    tcol = "Test_Loss" if "Test_Loss" in f.columns else f.columns[1]
    c_t = float(c["test_loss"].iloc[-1])
    c_tr = float(c["train_loss"].iloc[-1])
    f_t = float(f[tcol].iloc[-1])
    return c_t, c_tr, f_t


def main() -> None:
    task1_convergence()
    try:
        task2_traffic_sample()
    except Exception as e:
        print(f"Task 2 error: {e}", flush=True)
    c_test, _c_tr, f_test = _final_losses()
    print(
        f"Summary — Centralized test MSE: {c_test:.6f} | Federated test MSE: {f_test:.6f}",
        flush=True,
    )
    print("Experiment narrative: see README.md", flush=True)


if __name__ == "__main__":
    main()

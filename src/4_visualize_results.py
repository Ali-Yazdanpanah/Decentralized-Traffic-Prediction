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


def _set_plot_style() -> None:
    for st in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        if st in plt.style.available or st == "default":
            plt.style.use(st)
            break


def task3_partitioning_systems_curves() -> None:
    """
    Visualize partitioning ablation from systems run (fixed local epochs).
    """
    p = LOGS / "systems_partitioning_ablation.csv"
    if not p.is_file():
        print(f"Skip systems partitioning curve: missing {p}")
        return
    df = pd.read_csv(p)
    baseline = None
    cb = LOGS / "centralized_baseline.csv"
    if cb.is_file():
        c = pd.read_csv(cb)
        baseline = float(c["test_loss"].iloc[-1])

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    rounds = df["Round"].values
    ax.plot(
        rounds,
        df["Random_Partition_Test_Loss"].values,
        color="C1",
        linewidth=2.0,
        label="Random Partition",
    )
    ax.plot(
        rounds,
        df["Spectral_Min_Cut_Test_Loss"].values,
        color="C2",
        linewidth=2.0,
        label="Spectral (Min-Cut)",
    )
    ax.plot(
        rounds,
        df["Fluid_Balanced_Test_Loss"].values,
        color="C0",
        linewidth=2.0,
        label="Fluid (Balanced)",
    )
    if baseline is not None:
        ax.axhline(
            y=baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.8,
            label="Centralized baseline (test MSE target)",
        )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global test MSE")
    ax.set_title("Systems run: partitioning impact at fixed sync (3 epochs/round)")
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    out = FIGURES / "systems_partitioning_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def task4_sync_time_breakdown() -> None:
    """
    Stacked bar chart for systems sync strategies:
    data+transfer vs local training vs FedAvg sync.
    """
    p = LOGS / "systems_ablation_summary.csv"
    if not p.is_file():
        print(f"Skip sync time breakdown: missing {p}")
        return
    df = pd.read_csv(p)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    x = np.arange(len(df))
    w = 0.65
    data_t = df["Data_Load_Transfer_s"].values
    train_t = df["Local_Training_s"].values
    sync_t = df["FedAvg_Sync_s"].values
    ax.bar(x, data_t, width=w, color="C4", label="Data load + transfer")
    ax.bar(x, train_t, width=w, bottom=data_t, color="C0", label="Local training compute")
    ax.bar(x, sync_t, width=w, bottom=data_t + train_t, color="C3", label="FedAvg synchronization")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Strategy"].tolist(), rotation=12, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Systems timing breakdown by sync strategy")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    out = FIGURES / "systems_sync_timing_breakdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def task5_partition_timing_breakdown() -> None:
    """
    Compare timing by partition strategy (fixed local epochs in systems run).
    """
    p = LOGS / "systems_partitioning_ablation_timing.csv"
    if not p.is_file():
        print(f"Skip partition timing breakdown: missing {p}")
        return
    df = pd.read_csv(p)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    x = np.arange(len(df))
    w = 0.65
    data_t = df["Data_Load_Transfer_s"].values
    train_t = df["Local_Training_s"].values
    sync_t = df["FedAvg_Sync_s"].values
    ax.bar(x, data_t, width=w, color="C4", label="Data load + transfer")
    ax.bar(x, train_t, width=w, bottom=data_t, color="C0", label="Local training compute")
    ax.bar(x, sync_t, width=w, bottom=data_t + train_t, color="C3", label="FedAvg synchronization")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Strategy"].tolist(), rotation=12, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Systems timing breakdown by partition strategy (3 epochs/round)")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    out = FIGURES / "systems_partitioning_timing_breakdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def task6_systems_dashboard() -> None:
    """
    2x2 dashboard for system-focused ablation outputs.
    Panels:
      (1) Sync strategy loss vs elapsed time
      (2) Sync strategy timing breakdown
      (3) Partition strategy loss vs round (fixed local epochs)
      (4) Partition strategy timing breakdown
    """
    p_sync_curve = LOGS / "systems_ablation_roundwise.csv"
    p_sync_timing = LOGS / "systems_ablation_summary.csv"
    p_part_curve = LOGS / "systems_partitioning_ablation.csv"
    p_part_timing = LOGS / "systems_partitioning_ablation_timing.csv"
    missing = [
        str(p)
        for p in (p_sync_curve, p_sync_timing, p_part_curve, p_part_timing)
        if not p.is_file()
    ]
    if missing:
        print(f"Skip systems dashboard; missing: {', '.join(missing)}")
        return

    df_sync_curve = pd.read_csv(p_sync_curve)
    df_sync_timing = pd.read_csv(p_sync_timing)
    df_part_curve = pd.read_csv(p_part_curve)
    df_part_timing = pd.read_csv(p_part_timing)

    baseline = None
    cb = LOGS / "centralized_baseline.csv"
    if cb.is_file():
        c = pd.read_csv(cb)
        baseline = float(c["test_loss"].iloc[-1])

    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (1) sync loss vs elapsed time
    sync_colors = {
        "High Sync (1 epoch/round)": "C3",
        "Medium Sync (3 epochs/round)": "C0",
        "Low Sync (10 epochs/round)": "C2",
    }
    for strategy, dfi in df_sync_curve.groupby("Strategy"):
        dfi = dfi.sort_values("Round")
        ax1.plot(
            dfi["Elapsed_s"].values,
            dfi["Test_Loss"].values,
            label=strategy,
            linewidth=2.0,
            color=sync_colors.get(strategy, None),
        )
    if baseline is not None:
        ax1.axhline(
            y=baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            label="Centralized baseline",
        )
    ax1.set_title("A) Sync strategies: loss vs elapsed time")
    ax1.set_xlabel("Elapsed wall-clock time (s)")
    ax1.set_ylabel("Global test MSE")
    ax1.legend(fontsize=8, frameon=True, loc="best")

    # (2) sync timing breakdown
    x = np.arange(len(df_sync_timing))
    w = 0.6
    data_t = df_sync_timing["Data_Load_Transfer_s"].values
    train_t = df_sync_timing["Local_Training_s"].values
    sync_t = df_sync_timing["FedAvg_Sync_s"].values
    ax2.bar(x, data_t, width=w, color="C4", label="Data+transfer")
    ax2.bar(x, train_t, width=w, bottom=data_t, color="C0", label="Local compute")
    ax2.bar(x, sync_t, width=w, bottom=data_t + train_t, color="C3", label="FedAvg sync")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_sync_timing["Strategy"].tolist(), rotation=12, ha="right")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_title("B) Sync strategies: timing breakdown")
    ax2.legend(fontsize=8, frameon=True, loc="upper left")

    # (3) partition loss vs round
    rounds = df_part_curve["Round"].values
    ax3.plot(
        rounds,
        df_part_curve["Random_Partition_Test_Loss"].values,
        color="C1",
        linewidth=2.0,
        label="Random",
    )
    ax3.plot(
        rounds,
        df_part_curve["Spectral_Min_Cut_Test_Loss"].values,
        color="C2",
        linewidth=2.0,
        label="Spectral (Min-Cut)",
    )
    ax3.plot(
        rounds,
        df_part_curve["Fluid_Balanced_Test_Loss"].values,
        color="C0",
        linewidth=2.0,
        label="Fluid (Balanced)",
    )
    if baseline is not None:
        ax3.axhline(y=baseline, color="0.35", linestyle="--", linewidth=1.5, label="Centralized baseline")
    ax3.set_title("C) Partitions @ 3 epochs/round: loss vs round")
    ax3.set_xlabel("Communication round")
    ax3.set_ylabel("Global test MSE")
    ax3.legend(fontsize=8, frameon=True, loc="best")

    # (4) partition timing breakdown
    x2 = np.arange(len(df_part_timing))
    dt = df_part_timing["Data_Load_Transfer_s"].values
    tr = df_part_timing["Local_Training_s"].values
    sy = df_part_timing["FedAvg_Sync_s"].values
    ax4.bar(x2, dt, width=w, color="C4", label="Data+transfer")
    ax4.bar(x2, tr, width=w, bottom=dt, color="C0", label="Local compute")
    ax4.bar(x2, sy, width=w, bottom=dt + tr, color="C3", label="FedAvg sync")
    ax4.set_xticks(x2)
    ax4.set_xticklabels(df_part_timing["Strategy"].tolist(), rotation=12, ha="right")
    ax4.set_ylabel("Wall-clock time (s)")
    ax4.set_title("D) Partition strategies: timing breakdown")
    ax4.legend(fontsize=8, frameon=True, loc="upper left")

    fig.suptitle("Systems Dashboard: Federated STGNN (MPS run)", fontsize=14, y=0.995)
    fig.tight_layout()
    out = FIGURES / "systems_dashboard.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    task1_convergence()
    try:
        task2_traffic_sample()
    except Exception as e:
        print(f"Task 2 error: {e}", flush=True)
    task3_partitioning_systems_curves()
    task4_sync_time_breakdown()
    task5_partition_timing_breakdown()
    task6_systems_dashboard()
    c_test, _c_tr, f_test = _final_losses()
    print(
        f"Summary — Centralized test MSE: {c_test:.6f} | Federated test MSE: {f_test:.6f}",
        flush=True,
    )
    print("Experiment narrative: see README.md", flush=True)


if __name__ == "__main__":
    main()

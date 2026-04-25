"""
Systems efficiency ablation on Apple Silicon (MPS preferred, CPU fallback).

Runs three Fluid-partition FedAvg trials (10 rounds each) with different
communication intervals (local epochs per round): 1, 3, and 10.
Measures wall-clock time breakdowns and plots loss vs. elapsed time.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import time
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from networkx.algorithms import community
from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader, Subset


def _load_federated_module() -> ModuleType:
    p = Path(__file__).resolve().parent / "2_train_federated.py"
    spec = importlib.util.spec_from_file_location("train_federated", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fed = _load_federated_module()
CentralizedSTGNN = _fed.CentralizedSTGNN
TrafficWindowDataset = _fed.TrafficWindowDataset
_adj_to_edge_index = _fed._adj_to_edge_index
_minmax = _fed._minmax
_evaluate_full = _fed._evaluate_full
_fedavg = _fed._fedavg

WINDOW = _fed.WINDOW
HIDDEN_DIM = _fed.HIDDEN_DIM
BATCH_SIZE = _fed.BATCH_SIZE
LR = _fed.LR
N_ZONES = _fed.N_ZONES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"

TRIALS: list[tuple[str, int]] = [
    ("High Sync (1 epoch/round)", 1),
    ("Medium Sync (3 epochs/round)", 3),
    ("Low Sync (10 epochs/round)", 10),
]
N_ROUNDS = 10
PARTITION_LOCAL_EPOCHS = 3
SEED = 42
INIT_SEED = 11


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_graph(adj: np.ndarray, idx_to_str: dict[int, str]) -> nx.Graph:
    n = int(adj.shape[0])
    g = nx.Graph()
    for i in range(n):
        g.add_node(idx_to_str[i])
    for i in range(n):
        for j in range(i + 1, n):
            if int(adj[i, j]) == 1:
                g.add_edge(idx_to_str[i], idx_to_str[j])
    return g


def _fluid_zone_tensors(
    g: nx.Graph, idx_to_str: dict[int, str], name_to_idx: dict[str, int]
) -> list[torch.Tensor]:
    comms = list(community.asyn_fluidc(g, k=N_ZONES, seed=SEED))
    node_to_zone: dict[str, int] = {}
    for z, c in enumerate(comms):
        for u in c:
            node_to_zone[u] = z
    zones: list[list[int]] = [[] for _ in range(N_ZONES)]
    for i in range(len(idx_to_str)):
        node = idx_to_str[i]
        z = node_to_zone[node]
        zones[z].append(int(name_to_idx[node]))
    return [torch.tensor(sorted(z), dtype=torch.long) for z in zones]


def _zone_tensors_from_labels(
    labels: np.ndarray, idx_to_str: dict[int, str], name_to_idx: dict[str, int]
) -> list[torch.Tensor]:
    zones: list[list[int]] = [[] for _ in range(N_ZONES)]
    for i, z in enumerate(labels.tolist()):
        node = idx_to_str[i]
        zones[int(z)].append(int(name_to_idx[node]))
    return [torch.tensor(sorted(z), dtype=torch.long) for z in zones]


def _random_zone_tensors(
    idx_to_str: dict[int, str], name_to_idx: dict[str, int]
) -> list[torch.Tensor]:
    names = list(name_to_idx.keys())
    random.Random(SEED).shuffle(names)
    n = len(names)
    sizes = [n // N_ZONES + (1 if k < (n % N_ZONES) else 0) for k in range(N_ZONES)]
    name_to_zone: dict[str, int] = {}
    st = 0
    for z, sz in enumerate(sizes):
        for nm in names[st : st + sz]:
            name_to_zone[nm] = z
        st += sz
    labels = np.asarray([name_to_zone[idx_to_str[i]] for i in range(n)], dtype=np.int64)
    return _zone_tensors_from_labels(labels, idx_to_str, name_to_idx)


def _spectral_zone_tensors(
    adj: np.ndarray, idx_to_str: dict[int, str], name_to_idx: dict[str, int]
) -> list[torch.Tensor]:
    sc = SpectralClustering(
        n_clusters=N_ZONES,
        affinity="precomputed",
        random_state=SEED,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(adj).astype(np.int64)
    return _zone_tensors_from_labels(labels, idx_to_str, name_to_idx)


def _read_baseline_test_loss() -> float | None:
    p = LOGS / "centralized_baseline.csv"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["test_loss"])


def _train_one_client_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    edge_index: torch.Tensor,
    zone_idx: torch.Tensor,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """
    Returns:
      data_transfer_time, compute_train_time
    """
    model.train()
    data_t = 0.0
    comp_t = 0.0
    z = zone_idx.to(device)
    it = iter(train_loader)
    while True:
        t0 = time.perf_counter()
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()
        xb = xb.to(device)
        yb = yb.to(device)
        t2 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb, edge_index)
        pz = pred.index_select(1, z).index_select(2, z)
        yz = yb.index_select(1, z).index_select(2, z)
        loss = loss_fn(pz, yz)
        loss.backward()
        optimizer.step()
        t3 = time.perf_counter()

        data_t += (t1 - t0) + (t2 - t1)
        comp_t += (t3 - t2)
    return data_t, comp_t


def _run_trial(
    strategy_name: str,
    local_epochs: int,
    device: torch.device,
    n: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    zone_tensors: list[torch.Tensor],
) -> dict:
    torch.manual_seed(INIT_SEED)
    if device.type == "mps":
        torch.mps.manual_seed(INIT_SEED)

    global_model = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
    clients: list[nn.Module] = []
    opts: list[torch.optim.Optimizer] = []
    for _ in range(N_ZONES):
        m = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
        m.load_state_dict(global_model.state_dict())
        clients.append(m)
        opts.append(torch.optim.Adam(m.parameters(), lr=LR))

    loss_fn = nn.MSELoss()
    losses: list[float] = []
    elapsed_by_round: list[float] = []
    cumulative = 0.0

    data_transfer_time = 0.0
    local_training_time = 0.0
    sync_time = 0.0

    print(f"--- {strategy_name} ---", flush=True)
    for rnd in range(1, N_ROUNDS + 1):
        round_start = time.perf_counter()
        # broadcast global weights
        gstate = global_model.state_dict()
        for m in clients:
            m.load_state_dict(gstate)

        # local training
        for client_idx in range(N_ZONES):
            m = clients[client_idx]
            opt = opts[client_idx]
            z = zone_tensors[client_idx]
            for _ in range(local_epochs):
                dt, ct = _train_one_client_epoch(
                    m, opt, train_loader, edge_index, z, device, loss_fn
                )
                data_transfer_time += dt
                local_training_time += ct

        # sync + FedAvg
        ts0 = time.perf_counter()
        merged = _fedavg([m.state_dict() for m in clients], N_ZONES)
        global_model.load_state_dict(merged)
        ts1 = time.perf_counter()
        sync_time += ts1 - ts0

        test_loss = _evaluate_full(global_model, test_loader, edge_index, device, loss_fn)
        losses.append(test_loss)
        round_end = time.perf_counter()
        cumulative += round_end - round_start
        elapsed_by_round.append(cumulative)
        print(
            f"  Round {rnd:2d}/{N_ROUNDS} | loss={test_loss:.6f} | elapsed={cumulative:.2f}s",
            flush=True,
        )

    return {
        "strategy": strategy_name,
        "local_epochs": local_epochs,
        "losses": losses,
        "elapsed_by_round": elapsed_by_round,
        "final_mse": losses[-1],
        "total_time_s": cumulative,
        "data_transfer_s": data_transfer_time,
        "local_training_s": local_training_time,
        "sync_s": sync_time,
    }


def main() -> None:
    device = _select_device()
    print(f"Device selected: {device}", flush=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        name_to_idx: dict[str, int] = json.load(f)
    idx_to_str = {v: k for k, v in name_to_idx.items()}
    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    traffic = np.load(PROCESSED / "traffic_tensor.npy")
    n = int(adj.shape[0])
    t_total = int(traffic.shape[0])
    n_samples = t_total - WINDOW
    n_train = int(0.8 * n_samples)

    mm = PROCESSED / "traffic_minmax.json"
    if not mm.is_file():
        raise FileNotFoundError("Missing traffic_minmax.json. Run baseline prep/training first.")
    with open(mm, encoding="utf-8") as f:
        j = json.load(f)
    t_min, t_max = float(j["min"]), float(j["max"])
    traffic_n = _minmax(traffic, t_min, t_max).astype(np.float32)

    full_ds = TrafficWindowDataset(traffic_n, WINDOW)
    train_ds = Subset(full_ds, list(range(n_train)))
    test_ds = Subset(full_ds, list(range(n_train, n_samples)))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    edge_index = _adj_to_edge_index(adj).to(device)

    g = _to_graph(adj, idx_to_str)
    zone_tensors = _fluid_zone_tensors(g, idx_to_str, name_to_idx)
    zone_sizes = [int(z.numel()) for z in zone_tensors]
    print(f"Fluid zone sizes: {zone_sizes}", flush=True)

    results: list[dict] = []
    for strategy, local_epochs in TRIALS:
        results.append(
            _run_trial(
                strategy,
                local_epochs,
                device,
                n,
                train_loader,
                test_loader,
                edge_index,
                zone_tensors,
            )
        )

    baseline = _read_baseline_test_loss()
    # Estimated bandwidth savings vs 1-epoch sync frequency
    # syncs-per-local-update proxy = 1 / local_epochs
    for r in results:
        r["estimated_bw_savings_pct"] = 100.0 * (1.0 - 1.0 / float(r["local_epochs"]))

    detail_csv = LOGS / "systems_ablation_roundwise.csv"
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Strategy", "Local_Epochs", "Round", "Elapsed_s", "Test_Loss"])
        for r in results:
            for i, (t_s, loss) in enumerate(zip(r["elapsed_by_round"], r["losses"]), start=1):
                w.writerow([r["strategy"], r["local_epochs"], i, f"{t_s:.6f}", f"{loss:.8f}"])
    print(f"Saved: {detail_csv}", flush=True)

    summary_csv = LOGS / "systems_ablation_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Strategy",
                "Local_Epochs",
                "Final_MSE",
                "Total_Time_s",
                "Data_Load_Transfer_s",
                "Local_Training_s",
                "FedAvg_Sync_s",
                "Estimated_Bandwidth_Savings_pct",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["strategy"],
                    r["local_epochs"],
                    f"{r['final_mse']:.8f}",
                    f"{r['total_time_s']:.4f}",
                    f"{r['data_transfer_s']:.4f}",
                    f"{r['local_training_s']:.4f}",
                    f"{r['sync_s']:.4f}",
                    f"{r['estimated_bw_savings_pct']:.2f}",
                ]
            )
    print(f"Saved: {summary_csv}", flush=True)

    for st in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        if st in plt.style.available or st == "default":
            plt.style.use(st)
            break
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    colors = {
        "High Sync (1 epoch/round)": "C3",
        "Medium Sync (3 epochs/round)": "C0",
        "Low Sync (10 epochs/round)": "C2",
    }
    for r in results:
        ax.plot(
            r["elapsed_by_round"],
            r["losses"],
            linewidth=2.0,
            color=colors[r["strategy"]],
            label=r["strategy"],
        )
    if baseline is not None:
        ax.axhline(
            y=baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.8,
            label="Centralized baseline (test MSE target)",
        )
    ax.set_xlabel("Elapsed wall-clock time (s)")
    ax.set_ylabel("Global test MSE")
    ax.set_title("Systems ablation on Fluid partition: loss vs. time")
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    out_fig = FIGURES / "systems_ablation.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}", flush=True)

    print("\n=== Final Results ===", flush=True)
    for r in sorted(results, key=lambda x: x["total_time_s"]):
        print(
            f"{r['strategy']}: final_mse={r['final_mse']:.6f}, "
            f"time={r['total_time_s']:.2f}s, "
            f"BW_savings~{r['estimated_bw_savings_pct']:.1f}%",
            flush=True,
        )

    # Additional systems experiment: multiple partitioning at fixed sync interval.
    partition_trials = [
        ("Random Partition", _random_zone_tensors(idx_to_str, name_to_idx)),
        ("Spectral (Min-Cut)", _spectral_zone_tensors(adj, idx_to_str, name_to_idx)),
        ("Fluid (Balanced)", _fluid_zone_tensors(g, idx_to_str, name_to_idx)),
    ]
    print(
        f"\n=== Multi-partition systems trial @ {PARTITION_LOCAL_EPOCHS} local epochs/round ===",
        flush=True,
    )
    partition_results: list[dict] = []
    for name, zt in partition_trials:
        partition_results.append(
            _run_trial(
                name,
                PARTITION_LOCAL_EPOCHS,
                device,
                n,
                train_loader,
                test_loader,
                edge_index,
                zt,
            )
        )

    # CSV with same round-wise layout style as partitioning_ablation.csv
    part_csv = LOGS / "systems_partitioning_ablation.csv"
    by_name = {r["strategy"]: r for r in partition_results}
    with open(part_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Round",
                "Random_Partition_Test_Loss",
                "Spectral_Min_Cut_Test_Loss",
                "Fluid_Balanced_Test_Loss",
            ]
        )
        for i in range(N_ROUNDS):
            w.writerow(
                [
                    i + 1,
                    f"{by_name['Random Partition']['losses'][i]:.8f}",
                    f"{by_name['Spectral (Min-Cut)']['losses'][i]:.8f}",
                    f"{by_name['Fluid (Balanced)']['losses'][i]:.8f}",
                ]
            )
    print(f"Saved: {part_csv}", flush=True)

    # Extra timing detail CSV for systems context.
    part_timing_csv = LOGS / "systems_partitioning_ablation_timing.csv"
    with open(part_timing_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Strategy",
                "Local_Epochs",
                "Final_MSE",
                "Total_Time_s",
                "Data_Load_Transfer_s",
                "Local_Training_s",
                "FedAvg_Sync_s",
            ]
        )
        for r in partition_results:
            w.writerow(
                [
                    r["strategy"],
                    r["local_epochs"],
                    f"{r['final_mse']:.8f}",
                    f"{r['total_time_s']:.4f}",
                    f"{r['data_transfer_s']:.4f}",
                    f"{r['local_training_s']:.4f}",
                    f"{r['sync_s']:.4f}",
                ]
            )
    print(f"Saved: {part_timing_csv}", flush=True)


if __name__ == "__main__":
    main()

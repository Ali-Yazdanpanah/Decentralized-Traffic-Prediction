"""
Ablation: Random vs Spectral vs Fluid partitioning for FedAvg (10 rounds, 3 local epochs).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
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
    name = "train_federated"
    spec = importlib.util.spec_from_file_location(name, p)
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
_fit_minmax = _fed._fit_minmax
LocalClient = _fed.LocalClient
_fedavg = _fed._fedavg
_evaluate_full = _fed._evaluate_full
WINDOW = _fed.WINDOW
HIDDEN_DIM = _fed.HIDDEN_DIM
BATCH_SIZE = _fed.BATCH_SIZE
LR = _fed.LR
LOCAL_EPOCHS = _fed.LOCAL_EPOCHS
N_ZONES = _fed.N_ZONES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"

N_ROUNDS = 10
RANDOM_SEED = 42
AB_INIT_SEED = 7  # same for all 3 ablations to isolate partition effect
FOG_PATHS = {
    "random": PROCESSED / "fog_topology_random.json",
    "spectral": PROCESSED / "fog_topology_spectral.json",
    "fluid": PROCESSED / "fog_topology_fluid.json",
}


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


def _intra_zone_degree(
    g: nx.Graph, node: str, zone: int, zone_of: dict[str, int]
) -> int:
    return sum(1 for nb in g.neighbors(node) if zone_of[nb] == zone)


def _elect_aggregators(
    g: nx.Graph,
    idx_to_str: dict[int, str],
    zone_label: np.ndarray,
) -> dict[int, str]:
    n = len(idx_to_str)
    zone_of: dict[str, int] = {idx_to_str[i]: int(zone_label[i]) for i in range(n)}
    out: dict[int, str] = {}
    for z in range(N_ZONES):
        members = [idx_to_str[i] for i in range(n) if int(zone_label[i]) == z]
        if not members:
            print(f"Warning: zone {z} empty.", file=sys.stderr)
            continue
        best: str | None = None
        best_s = -1
        for node in sorted(members):
            s = _intra_zone_degree(g, node, z, zone_of)
            if s > best_s:
                best_s = s
                best = node
            elif s == best_s and best is not None and node < best:
                best = node
        if best is not None:
            out[z] = best
    return out


def _fog_to_json(
    idx_to_str: dict[int, str],
    zone_label: np.ndarray,
    ag: dict[int, str],
) -> dict[str, dict[str, list[str] | str]]:
    n = len(idx_to_str)
    out: dict = {}
    for z in range(N_ZONES):
        nodes = sorted(
            [idx_to_str[i] for i in range(n) if int(zone_label[i]) == z]
        )
        out[f"zone_{z}"] = {"nodes": nodes, "aggregator": ag.get(z, "")}
    return out


def _zone_tensors_from_fog(
    fog: dict, name_to_id: dict[str, int]
) -> list[torch.Tensor]:
    zt: list[torch.Tensor] = []
    for z in range(N_ZONES):
        names: list = fog[f"zone_{z}"]["nodes"]
        idx = sorted(int(name_to_id[nm]) for nm in names)
        zt.append(torch.tensor(idx, dtype=torch.long))
    return zt


def partition_random(
    n: int, g: nx.Graph, idx_to_str: dict[int, str], name_to_id: dict[str, int]
) -> dict:
    all_names = list(name_to_id.keys())
    random.Random(RANDOM_SEED).shuffle(all_names)
    sizes = [
        n // N_ZONES + (1 if j < (n % N_ZONES) else 0) for j in range(N_ZONES)
    ]
    i0 = 0
    name_to_z: dict[str, int] = {}
    for z, sz in enumerate(sizes):
        for nm in all_names[i0 : i0 + sz]:
            name_to_z[nm] = z
        i0 += sz
    zl = np.array(
        [name_to_z[idx_to_str[i]] for i in range(n)], dtype=np.int64
    )
    ag = _elect_aggregators(g, idx_to_str, zl)
    return _fog_to_json(idx_to_str, zl, ag)


def partition_spectral(adj: np.ndarray, g: nx.Graph, idx_to_str: dict[int, str]) -> dict:
    sc = SpectralClustering(
        n_clusters=N_ZONES,
        affinity="precomputed",
        random_state=RANDOM_SEED,
        assign_labels="kmeans",
    )
    zl = sc.fit_predict(adj).astype(np.int64)
    ag = _elect_aggregators(g, idx_to_str, zl)
    return _fog_to_json(idx_to_str, zl, ag)


def partition_fluid(
    g: nx.Graph, idx_to_str: dict[int, str]
) -> dict:
    comms = list(community.asyn_fluidc(g, k=N_ZONES, seed=RANDOM_SEED))
    n = len(idx_to_str)
    node_to_z: dict[str, int] = {}
    for z, c in enumerate(comms):
        for u in c:
            node_to_z[u] = z
    zl = np.array(
        [node_to_z[idx_to_str[i]] for i in range(n)], dtype=np.int64
    )
    ag = _elect_aggregators(g, idx_to_str, zl)
    return _fog_to_json(idx_to_str, zl, ag)


def _run_tournament(
    device: torch.device,
    n: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    fog: dict,
) -> list[float]:
    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        name_to_id: dict = json.load(f)
    zt = _zone_tensors_from_fog(fog, name_to_id)
    mse = nn.MSELoss()
    torch.manual_seed(AB_INIT_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(AB_INIT_SEED)
    global_m = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
    clients: list[LocalClient] = []
    for z in range(N_ZONES):
        c = CentralizedSTGNN(n, HIDDEN_DIM).to(device)
        c.load_state_dict(global_m.state_dict())
        clients.append(LocalClient(z, zt[z].clone().cpu(), c))
    history: list[float] = []
    for _r in range(1, N_ROUNDS + 1):
        gstate = global_m.state_dict()
        for c in clients:
            c.set_weights(gstate)
        for c in clients:
            c.train_one_round(train_loader, edge_index, device, mse)
        global_m.load_state_dict(_fedavg([c.get_state() for c in clients], N_ZONES))
        te = _evaluate_full(
            global_m, test_loader, edge_index, device, mse
        )
        history.append(te)
    return history


def _read_centralized_test_baseline() -> float | None:
    p = LOGS / "centralized_baseline.csv"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["test_loss"])


def _zone_sizes(fog: dict) -> list[int]:
    return [len(fog[f"zone_{z}"]["nodes"]) for z in range(N_ZONES)]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | {N_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs", flush=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        name_to_id: dict = json.load(f)
    idx_to_str = {v: k for k, v in name_to_id.items()}
    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    traffic = np.load(PROCESSED / "traffic_tensor.npy")
    n = int(adj.shape[0])
    t_total = int(traffic.shape[0])
    w = WINDOW
    n_samples = t_total - w
    n_train = int(0.8 * n_samples)
    g = _to_graph(adj, idx_to_str)

    mm = PROCESSED / "traffic_minmax.json"
    if not mm.is_file():
        raise FileNotFoundError("traffic_minmax.json required; run 1_prepare_data and baseline first.")
    with open(mm, encoding="utf-8") as f:
        jm = json.load(f)
    t_min, t_max = float(jm["min"]), float(jm["max"])
    traffic_n = _minmax(traffic, t_min, t_max).astype(np.float32)
    full_ds = TrafficWindowDataset(traffic_n, w)
    train_ds = Subset(full_ds, list(range(n_train)))
    test_ds = Subset(full_ds, list(range(n_train, n_samples)))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    edge_index = _adj_to_edge_index(adj).to(device)

    fogs = {
        "Random Partition": (
            "random",
            partition_random(n, g, idx_to_str, name_to_id),
        ),
        "Spectral (Min-Cut)": (
            "spectral",
            partition_spectral(adj, g, idx_to_str),
        ),
        "Fluid (Balanced)": (
            "fluid",
            partition_fluid(g, idx_to_str),
        ),
    }

    for _label, (key, fd) in fogs.items():
        p = FOG_PATHS[key]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(fd, f, indent=2, sort_keys=True, ensure_ascii=True)
        print(f"Wrote {p} | zone sizes: {_zone_sizes(fd)}", flush=True)

    curves: dict[str, list[float]] = {}
    for label, (_key, fd) in fogs.items():
        print(f"--- Ablation: {label} ---", flush=True)
        torch.manual_seed(AB_INIT_SEED)  # reset for identical init
        h = _run_tournament(
            device, n, train_loader, test_loader, edge_index, fd
        )
        curves[label] = h
        for i, y in enumerate(h, 1):
            print(f"  R{i:2d} test: {y:.6f}", flush=True)

    baseline = _read_centralized_test_baseline()
    rounds_axis = list(range(1, N_ROUNDS + 1))
    for st in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        if st in plt.style.available or st == "default":
            plt.style.use(st)
            break
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    colors = {
        "Random Partition": "C1",
        "Spectral (Min-Cut)": "C2",
        "Fluid (Balanced)": "C0",
    }
    styles = {
        "Random Partition": "-",
        "Spectral (Min-Cut)": "-",
        "Fluid (Balanced)": "-",
    }
    for label, h in curves.items():
        ax.plot(
            rounds_axis,
            h,
            linestyle=styles[label],
            color=colors[label],
            label=label,
            linewidth=2.0,
        )
    if baseline is not None:
        ax.axhline(
            y=baseline,
            color="0.35",
            linestyle="--",
            linewidth=1.8,
            label="Centralized baseline (test MSE, target)",
        )
    ax.set_xlabel("Communication round", fontsize=11)
    ax.set_ylabel("Global test MSE (full matrix)", fontsize=11)
    ax.set_title("Partitioning ablation: Random vs. Spectral vs. Fluid", fontsize=12)
    ax.set_xticks(range(0, N_ROUNDS + 1, 2))
    ax.legend(frameon=True, loc="best", fontsize=9)
    fig.tight_layout()
    out_png = FIGURES / "partitioning_ablation.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}", flush=True)

    csv_p = LOGS / "partitioning_ablation.csv"
    with open(csv_p, "w", encoding="utf-8", newline="") as f:
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
                    f"{curves['Random Partition'][i]:.8f}",
                    f"{curves['Spectral (Min-Cut)'][i]:.8f}",
                    f"{curves['Fluid (Balanced)'][i]:.8f}",
                ]
            )
    print(f"Saved: {csv_p}", flush=True)

    finals = {k: v[-1] for k, v in curves.items()}
    print("\n=== Final test MSE (round %d) ===" % N_ROUNDS, flush=True)
    for k, v in sorted(finals.items(), key=lambda x: x[1]):
        print(f"  {k}: {v:.6f}", flush=True)
    if baseline is not None:
        print("\n=== Convergence gap vs. centralized test MSE (final round) ===", flush=True)
        gaps = {k: v - baseline for k, v in finals.items()}
        for k, g2 in sorted(gaps.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {g2:+.6f} (MSE - baseline)", flush=True)
        spread = max(gaps.values()) - min(gaps.values())
        print(
            f"\nSpread in final excess loss (max gap − min gap): {spread:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

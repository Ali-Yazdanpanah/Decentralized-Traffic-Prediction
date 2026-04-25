"""
Partition the GEANT graph into fog zones and elect per-zone federated aggregators.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures"
N_CLUSTERS = 4
RANDOM_STATE = 42


def _load_id_maps() -> tuple[dict[str, int], dict[int, str], np.ndarray]:
    with open(PROCESSED / "node_mapping.json", encoding="utf-8") as f:
        str_to_idx: dict[str, int] = json.load(f)
    idx_to_str: dict[int, str] = {v: k for k, v in str_to_idx.items()}
    adj = np.load(PROCESSED / "adjacency_matrix.npy")
    return str_to_idx, idx_to_str, adj


def _to_networkx(adj: np.ndarray, idx_to_str: dict[int, str]) -> nx.Graph:
    n = int(adj.shape[0])
    g = nx.Graph()
    for i in range(n):
        g.add_node(idx_to_str[i])
    for i in range(n):
        for j in range(i + 1, n):
            if int(adj[i, j]) == 1:
                g.add_edge(idx_to_str[i], idx_to_str[j])
    return g


def _fluid_zones(g: nx.Graph, idx_to_str: dict[int, str]) -> np.ndarray:
    communities = list(community.asyn_fluidc(g, k=N_CLUSTERS, seed=RANDOM_STATE))
    n = len(idx_to_str)
    if len(communities) != N_CLUSTERS:
        print(
            f"Warning: expected {N_CLUSTERS} communities, got {len(communities)}.",
            file=sys.stderr,
        )
    node_to_zone: dict[str, int] = {}
    for zone_id, comm in enumerate(communities):
        for u in comm:
            node_to_zone[u] = zone_id
    for i in range(n):
        if idx_to_str[i] not in node_to_zone:
            raise KeyError(
                f"Node {idx_to_str[i]!r} not assigned a zone by asyn_fluidc; "
                "check that the graph and k match."
            )
    return np.asarray([node_to_zone[idx_to_str[i]] for i in range(n)], dtype=np.int64)


def _intra_zone_degree(
    g: nx.Graph, node: str, zone: int, zone_of: dict[str, int]
) -> int:
    c = 0
    for nb in g.neighbors(node):
        if zone_of[nb] == zone:
            c += 1
    return c


def _elect_aggregators(
    g: nx.Graph,
    idx_to_str: dict[int, str],
    zone_label: np.ndarray,
) -> tuple[dict[str, int], dict[int, str]]:
    n = len(idx_to_str)
    zone_of: dict[str, int] = {idx_to_str[i]: int(zone_label[i]) for i in range(n)}

    aggregators: dict[int, str] = {}
    for z in range(N_CLUSTERS):
        members = [idx_to_str[i] for i in range(n) if zone_label[i] == z]
        if not members:
            print(f"Warning: zone {z} is empty; skipping aggregator.", file=sys.stderr)
            continue
        best_node: str | None = None
        best_score = -1
        for node in sorted(members):
            s = _intra_zone_degree(g, node, z, zone_of)
            if s > best_score:
                best_score = s
                best_node = node
            elif s == best_score and best_node is not None and node < best_node:
                best_node = node
        if best_node is not None:
            aggregators[z] = best_node
    return zone_of, aggregators


def _build_fog_dict(
    idx_to_str: dict[int, str],
    zone_label: np.ndarray,
    aggregators: dict[int, str],
) -> dict[str, dict[str, list[str] | str]]:
    n = len(idx_to_str)
    out: dict[str, dict[str, list[str] | str]] = {}
    for z in range(N_CLUSTERS):
        nodes = sorted(
            [idx_to_str[i] for i in range(n) if zone_label[i] == z],
            key=lambda s: s,
        )
        agg = aggregators.get(z, "")
        out[f"zone_{z}"] = {"nodes": nodes, "aggregator": agg}
    return out


def _plot_fog(
    g: nx.Graph,
    zone_of: dict[str, int],
    aggregators: dict[int, str],
    out_path: Path,
) -> None:
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(g, seed=RANDOM_STATE, k=1.2 / (len(g) ** 0.5))
    colors = [plt.cm.tab10(zone_of[n] % 10) for n in g.nodes()]

    agg_set = {aggregators.get(z) for z in range(N_CLUSTERS) if aggregators.get(z)}
    normal_nodes = [n for n in g.nodes() if n not in agg_set]
    ax_nodes = [n for n in g.nodes() if n in agg_set]

    if normal_nodes:
        nc = [zone_of[n] for n in normal_nodes]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=normal_nodes,
            node_color=nc,
            cmap=plt.cm.tab10,
            node_size=380,
            vmin=0,
            vmax=N_CLUSTERS - 1,
            edgecolors="gray",
            linewidths=0.5,
        )
    for agg in ax_nodes:
        z = zone_of[agg]
        c = [z]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[agg],
            node_color=c,
            cmap=plt.cm.tab10,
            node_size=650,
            vmin=0,
            vmax=N_CLUSTERS - 1,
            edgecolors="crimson",
            linewidths=2.5,
        )
    nx.draw_networkx_edges(g, pos, width=0.6, alpha=0.5, edge_color="0.45")
    nx.draw_networkx_labels(g, pos, font_size=6)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    _, idx_to_str, adj = _load_id_maps()
    n = int(adj.shape[0])
    g = _to_networkx(adj, idx_to_str)
    if g.number_of_edges() == 0:
        print("Warning: graph has no edges; clustering may be arbitrary.", file=sys.stderr)

    zone_label = _fluid_zones(g, idx_to_str)
    zone_of, aggregators = _elect_aggregators(g, idx_to_str, zone_label)
    fog = _build_fog_dict(idx_to_str, zone_label, aggregators)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "fog_topology.json", "w", encoding="utf-8") as f:
        json.dump(fog, f, indent=2, ensure_ascii=True, sort_keys=True)

    fig_path = FIGURES / "fog_topology.png"
    _plot_fog(g, zone_of, aggregators, fig_path)
    print(f"Saved: {PROCESSED / 'fog_topology.json'}")
    print(f"Saved: {fig_path}\n")
    sizes: list[int] = []
    for z in range(N_CLUSTERS):
        name = f"zone_{z}"
        info = fog.get(name, {})
        nodes: list = list(info.get("nodes", []))
        sizes.append(len(nodes))
        agg: str = str(info.get("aggregator", ""))
        print(f"{name}: {len(nodes)} nodes | aggregator: {agg}")
        print(f"  nodes: {', '.join(nodes)}")
    smin, smax = min(sizes), max(sizes)
    print(
        f"\nZone sizes: {sizes}  (min={smin}, max={smax}, spread={smax - smin})"
    )


if __name__ == "__main__":
    main()

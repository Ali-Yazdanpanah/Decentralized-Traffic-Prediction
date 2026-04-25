"""
Build adjacency from SNDlib native topology.txt and traffic from demand XML.

topology.txt: NODES ( ... ) and LINKS ( ... ) in the GÉANT Uhlig drop folder.
Demand snapshots: demandMatrix-*.xml with snd:demand entries.
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ns: dict[str, str] = {"snd": "http://sndlib.zib.de/network"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "directed-geant-uhlig-15min-over-4months-ALL"
)
TOPOLOGY_FILE = DATA_DIR / "topology.txt"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def _load_topology_text(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing topology file: {path}")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _parse_nodes_block(lines: list[str]) -> list[str]:
    """Return node IDs in file order from NODES ( ... )."""
    i = 0
    while i < len(lines) and "NODES" not in lines[i]:
        i += 1
    if i >= len(lines):
        raise ValueError("NODES ( block not found in topology.txt")
    # Line is like: NODES (
    i += 1
    ids: list[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == ")" or stripped == ");":
            break
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        first = stripped.split()[0]
        ids.append(first)
        i += 1
    if not ids:
        raise ValueError("No node IDs parsed from NODES block")
    return ids


def _parse_link_line_source_target(line: str) -> tuple[str, str] | None:
    """
    Line format: link_id ( source target ) ... optional ( numbers ... )
    Return the first ( a b ) pair where a,b are not both numeric.
    """
    for m in re.finditer(r"\(\s*(\S+)\s+(\S+)\s*\)", line):
        a, b = m.group(1), m.group(2)
        try:
            float(a)
            float(b)
        except ValueError:
            return a, b
    return None


def _parse_links_block(lines: list[str]) -> list[tuple[str, str]]:
    """Return (source, target) for each link from LINKS ( ... )."""
    i = 0
    while i < len(lines) and "LINKS" not in lines[i]:
        i += 1
    if i >= len(lines):
        raise ValueError("LINKS ( block not found in topology.txt")
    i += 1
    pairs: list[tuple[str, str]] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == ")" or stripped == ");":
            break
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        pr = _parse_link_line_source_target(line)
        if pr is not None:
            pairs.append(pr)
        i += 1
    return pairs


def _topology_from_file(path: Path) -> tuple[dict[str, int], np.ndarray, int]:
    """
    node_to_idx in NODES file order, symmetric 0/1 adjacency, and
    number of link rows in the text file (physical undirected links).
    """
    lines = _load_topology_text(path)
    node_ids = _parse_nodes_block(lines)
    node_to_idx: dict[str, int] = {nid: j for j, nid in enumerate(node_ids)}
    n = len(node_to_idx)
    adj = np.zeros((n, n), dtype=np.int8)
    link_pairs = _parse_links_block(lines)
    n_links = len(link_pairs)
    for a, b in link_pairs:
        if a not in node_to_idx or b not in node_to_idx:
            print(
                f"Warning: link {a!r} -> {b!r} references unknown node; skipping",
                file=sys.stderr,
            )
            continue
        ia, ib = node_to_idx[a], node_to_idx[b]
        adj[ia, ib] = 1
        adj[ib, ia] = 1
    return node_to_idx, adj, n_links


def _fill_traffic(
    all_xml: list[Path], node_to_idx: dict[str, int], t_shape: int, n: int
) -> tuple[np.ndarray, int]:
    traffic = np.zeros((t_shape, n, n), dtype=np.float64)
    skipped = 0
    for t, path in enumerate(all_xml):
        root = ET.parse(path).getroot()
        for dem in root.findall(".//snd:demand", ns):
            s_el = dem.find("snd:source", ns)
            d_el = dem.find("snd:target", ns)
            v_el = dem.find("snd:demandValue", ns)
            if s_el is None or d_el is None or v_el is None:
                continue
            if s_el.text is None or d_el.text is None or v_el.text is None:
                continue
            a, b = s_el.text.strip(), d_el.text.strip()
            if a not in node_to_idx or b not in node_to_idx:
                skipped += 1
                continue
            i, j = node_to_idx[a], node_to_idx[b]
            traffic[t, i, j] = float(v_el.text.strip())
    return traffic, skipped


def _clip_p99_nonzero(traffic: np.ndarray) -> tuple[np.ndarray, float]:
    pos = traffic[traffic > 0]
    if pos.size == 0:
        return traffic, 0.0
    p99 = float(np.percentile(pos, 99))
    clipped = np.clip(traffic, 0.0, p99)
    return clipped, p99


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print(f"Loading topology: {TOPOLOGY_FILE.name}")
    node_to_idx, adj_matrix, n_physical_links = _topology_from_file(TOPOLOGY_FILE)
    n = len(node_to_idx)

    all_xml = sorted(DATA_DIR.glob("*.xml"))
    if not all_xml:
        raise FileNotFoundError(f"No *.xml demand files under {DATA_DIR}")
    t_count = len(all_xml)

    traffic_tensor, n_skipped = _fill_traffic(
        all_xml, node_to_idx, t_count, n
    )
    if n_skipped:
        print(
            f"Warning: skipped {n_skipped} demand(s) with unknown node id(s).",
            file=sys.stderr,
        )

    traffic_tensor, p99_threshold = _clip_p99_nonzero(traffic_tensor)

    with open(OUT_DIR / "node_mapping.json", "w", encoding="utf-8") as f:
        json.dump(node_to_idx, f, indent=2, ensure_ascii=True, sort_keys=False)
    np.save(OUT_DIR / "adjacency_matrix.npy", adj_matrix)
    np.save(OUT_DIR / "traffic_tensor.npy", traffic_tensor)

    assert int(np.count_nonzero(np.triu(adj_matrix, 1) > 0)) == n_physical_links
    print("---")
    print(f"Number of nodes: {n}")
    print(
        f"Number of physical undirected edges (LINKS lines in topology.txt): "
        f"{n_physical_links}"
    )
    print(f"Final traffic tensor shape (T, N, N): {traffic_tensor.shape}")
    print(
        f"99th-percentile (non-zero traffic) clipping threshold: {p99_threshold:.6f}"
    )


if __name__ == "__main__":
    main()

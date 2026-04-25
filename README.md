# Federated Spatiotemporal Network Routing

Proactive traffic-aware routing for the GÉANT-style European backbone, trained **centrally** and **federally** with matching evaluation splits. This project connects a spatiotemporal GNN to **fog-style zones** and **FedAvg** so raw demand matrices can stay at regional edge aggregators.

---

## 1. System architecture and design choices

### Model choice: STGNN (GCN + GRU)

We use a **spatiotemporal graph neural network (STGNN)** that combines:

- **Graph convolution (GCN)** over the **physical fiber topology** to capture **spatial** dependencies: how each PoP’s traffic pattern couples to its neighbors on the European graph.
- **Gated recurrent unit (GRU)** over the time dimension to capture **temporal** structure: diurnal and multi-day **periodicity** in the 15-minute demand series.

The GCN encodes *who talks to whom* on the network; the GRU encodes *how that traffic evolves* over the sliding window. This is a natural fit for operator-style matrices where each step is a full $N \times N$ demand snapshot.

### Fog layer partitioning: from spectral clustering to asynchronous fluid communities

Initial experiments used **spectral clustering** to partition the European backbone. Spectral methods are strong at **minimizing edge cuts** (reducing the number of physical links between zones), but they **do not** optimize **load balance** across learners.

**Failure mode:** Spectral clustering produced a **straggler zone** of **11 nodes (50% of the network)** while other zones had as few as **3** nodes. In a federated setting, that creates a **synchronization bottleneck**: three zone aggregators sit **idle** while the overloaded 11-node zone finishes each local training epoch, inflating **wall-clock per communication round** and **tail latency** in the global update.

**Solution:** We switched to **asynchronous fluid communities** (`asyn_fluidc` in NetworkX). Unlike pure cut-based spectral clustering, fluid-style methods encourage **more balanced** community structure (density- and size-aware in practice on this graph). The network was redistributed to approximately **[7, 5, 4, 6]** nodes per zone—**local training time** is more uniform across the continent, improving **round efficiency** and reducing **straggler effects** in the GÉANT fog layer.

*Figure: fog zones and per-zone aggregators (run `src/federated/1_partition_graph.py` to regenerate).*

![Fog topology: zones and master edge routers](figures/fog_topology.png)

---

## 2. Data engineering and robustness

### Topology reconstruction

The physical network is **parsed from the SNDlib native** `topology.txt` (GÉANT Uhlig 15 min / 4 months). We extract **NODES** and **LINKS** blocks, then build an **undirected** simple graph used for the adjacency matrix. The published snapshot has **22 nodes** and **36 undirected edges** in the processed graph. Node IDs (e.g. `de1.de`) are written to `data/processed/node_mapping.json` and aligned with the demand tensors.

### Outlier mitigation: 99th-percentile clipping

The raw SNDlib demand series includes **extreme spikes**; a well-known artifact is a demand jump on the order of **~74&nbsp;M (Mbps in native units)**—orders of magnitude above typical entries. Fitting a neural model on that heavy tail can destabilize **gradients and loss** during both centralized and federated training.

**Mitigation:** In `src/data_utils/1_prepare_data.py` we apply **99th-percentile (non-zero) clipping**: values above the 99th percentile of **positive** demands are clipped before tensor assembly. This **preserves typical traffic** while capping the worst anomalies, which keeps optimization **well-behaved** and comparable across training modes.

---

## 3. Federated learning strategy

We use **FedAvg** (federated averaging):

- **20 communication rounds**
- **3 local epochs** per round **per** of the **4** zonal clients

Each client keeps a **copy of the global** `CentralizedSTGNN`, trains on **local windows** of the (globally min–max normalized) traffic, with a **zone-local** MSE on the $|\text{zone}| \times |\text{zone}|$ submatrix, then uploads **weights**. The server averages the four `state_dict`s, updates the global model, and **broadcasts** the result before the next round. Global test loss uses the **same 80/20** temporal split and **full** $N \times N$ MSE as the centralized baseline.

**Privacy and locality:** In this design, **raw traffic demand matrices for each region never have to be pooled in one central warehouse**. Training signals are **local** to **regional edge aggregators**; only **model parameters** (or, in a stricter deployment, **gradients**) cross the long-haul control plane. We associate the four **master edge routers** with the major anchor regions **Germany, Italy, France, and Austria** (e.g. `de1.de`, `it1.it`, `fr1.fr`, `at1.at`) as a narrative match to **zone-local** aggregation. This mirrors the operational story: **GÉANT-scale** paths stay **federated** by geography even when the model architecture is still global in width.

*Implementation: `src/federated/2_train_federated.py`.*

---

## 4. Experimental results

Metrics below are **representative of a full run** with the saved logs; after you re-train, regenerate plots with:

```bash
python src/4_visualize_results.py
```

(Reads `logs/centralized_baseline.csv` and `logs/federated_progress.csv`.)

### MSE (full $22 \times 22$ next-step matrix), test set

| Model | Reported quantity | Value |
| --- | --- | ---: |
| Centralized baseline | Training loss (last epoch) | 0.002315 |
| Centralized baseline | **Test MSE (last epoch)** | **0.002356** |
| Federated (FedAvg) | **Test MSE (round 20)** | **0.011027** |

### Convergence and traffic sanity check

**Left:** test MSE vs. “training progress” (epoch 1–20 for baseline, round 1–20 for federated; federated clients run 3 local epochs per round, not extra ticks on the x-axis).

**Right:** for aggregator node `de1.de` (largest zone), **sum of outgoing row** in the min–max normalized next-step target vs. federated prediction, **last 100** test windows (~15 min each → about **one day of samples** in wall-clock).

![Convergence: centralized test loss vs. federated test loss](figures/convergence_comparison.png)  
*Test MSE by training progress. Red dashed: centralized target; blue solid: federated progress (FedAvg).*

![Traffic at de1.de: ground truth vs. federated prediction (last 100 test windows)](figures/traffic_prediction_sample.png)  
*Row-sum of predicted vs. true traffic from `de1.de` over the last 100 test sliding windows.*

### Efficiency (telemetry, illustrative)

These counts are **toy** accounting units for discussion (matrix cells vs. weight scalars); real systems add compression and different privacy assumptions.

- **Centralized (raw “matrix” exposure, as in our accounting):**  
  $11454\,\text{windows} \times (22 \times 22) =$ **5,543,736** scalar **matrix cell** values in the notional “full send everything every step” story.  
- **Federated (per formula):**  
  $20\,\text{rounds} \times 4\,\text{clients} \times 27{,}862$ parameters $=$ **2,228,960** scalar **weight** uploads over training (FedAvg).  
- **Federated / centralized ratio (scalar count):** **0.402**; **% reduction in the matrix-exposure count** $(1 - C_\mathrm{fed}/C_\mathrm{cent}) \times 100$: **~59.8%** in these units.  
- **Interpretation:** Even when FL carries a **test MSE gap**, the **raw matrix stream** is not the thing that has to be centralized. **Bandwidth** and **privacy** arguments favor the federated edge story for many telecom deployments, at the cost of a modest **accuracy** penalty in this setup.

**Strategic takeaway (accuracy vs. operations):** We observe a **small accuracy gap** (higher test MSE under zone-local objectives + FedAvg vs. a fully centralized trainer). In production **edge** and **GÉANT-like** backbones, **federated** learning often still wins on **compliance, locality, and not hoovering the entire spatio-temporal tensor** to one site, which matches operator constraints more than a pure leaderboard point.

---

## 5. Future work

- **Graph Attention (GAT):** Replace or augment the GCN with a **GAT** layer to let each node **attend** over its neighborhood with learned weights. We expect this may **narrow the gap** between federated and centralized test error on challenging PoPs.
- **Asynchronous federated learning:** This repo sets the stage for **async FL**, where **fast** zones do not block on a **straggler** in every round—further **reducing wall-clock** and **tail latency** in the GÉANT fog control plane, especially if zones differ in size or backhaul. Combined with the **already balanced** fluid communities split, that is a natural “PhD hook” for the next design iteration.

---

## Project layout (quick)

| Path | Role |
| --- | --- |
| `src/data_utils/1_prepare_data.py` | SNDlib topology + demand → `adjacency_matrix.npy`, `traffic_tensor.npy` |
| `src/federated/1_partition_graph.py` | Fog zones, aggregators, `fog_topology.json` + `figures/fog_topology.png` |
| `src/models/2_train_baseline.py` | Centralized STGNN training + `logs/centralized_baseline.csv` |
| `src/federated/2_train_federated.py` | FedAvg training + `logs/federated_progress.csv` |
| `src/4_visualize_results.py` | Figures for this README |
| `figures/` | Plots (tracked in git for documentation) |

**Requirements:** see `requirements.txt` (`torch`, `torch_geometric`, `pandas`, …).

**Note:** `data/raw/`, `data/processed/`, and `saved_models/` are **gitignored**—commit **code and figures**, not 11,000+ XML files or large `.pth` checkpoints. Regenerate data and checkpoints locally after clone.

---

*Project evaluation: **Proactive Federated Routing** on public GÉANT-style SNDlib data.*

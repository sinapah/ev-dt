# src/ — Simulation Modules

| File | Purpose |
|------|---------|
| `config.py` | Constants: charger counts (Caltech=30, JPL=32), FL aggregation interval (24h), feature lag specs, column name mappings |
| `data_loader.py` | Loads `ground_truth.csv`, pivots from long (one row per site-hour) to wide format, maps CSV column names to internal names, fills NaN |
| `environment.py` | Iterates through ground truth timestamps, provides per-site state (arrivals, queue, wait, utilization, service time) and total incoming demand |
| `model.py` | `ForecastingModel` — sklearn LinearRegression wrapper with `get_weights()`/`set_weights()` for Federated Averaging |
| `edge_node.py` | `EdgeNode` — stores local history, engineers features (hour, day-of-week, arrival/queue/utilization lags), trains local model, predicts next-hour arrivals |
| `federated_coordinator.py` | `FederatedCoordinator` — collects model weights from edge nodes, performs FedAvg, distributes global model back |
| `digital_twin.py` | `DigitalTwin` — receives current state + predicted arrivals, predicts future queue/wait/utilization/congestion per site using a simple queue approximation |
| `queue_simulator.py` | `QueueSimulator` — lightweight discrete queue dynamics: given routed arrivals and service rates, computes resulting queue length, waiting time, and utilization |
| `scheduler.py` | `Scheduler` — see [Routing: Baseline vs DT-Guided](#routing-baseline-vs-dt-guided) below |
| `simulation_runner.py` | `SimulationRunner` — orchestrates the two-phase workflow: **training** (FL every 24h) then **evaluation** (DT vs baseline comparison with queue simulation) |
| `main.py` | Entry point — runs full pipeline and saves results to CSV |

## Routing: Baseline vs DT-Guided

Both modes in `Scheduler` (`scheduler.py`) share a two-stage structure:

1. **Captive split** — 60% of arrivals at each site must charge at their natural site (configurable via `CAPTIVE_FRACTION`).
2. **Flexible split** — the remaining 40% can be routed to either Caltech or JPL.

| | Baseline | DT-Guided |
|---|---|---|
| **Input** | Static `historical_split` (historical ratio of Caltech arrivals to total demand) | Predicted waiting times from the Digital Twin's what-if simulation |
| **Logic** | Flexible EVs split proportionally to the historical ratio: `flex_cal = total_flexible * historical_split` | Flexible EVs split by inverse-waiting-time weighting: `share_site = (1/wt_site) / sum(1/wt_all_sites)` |
| **Adaptivity** | None — a fixed 40.4% / 59.6% split regardless of current congestion | Dynamic — buses more flexible traffic toward the site with shorter predicted queues |

### Why DT-guided is more efficient

The Digital Twin (`digital_twin.py`) runs a fast what-if simulation using current queue state and predicted arrivals (from EdgeNodes) to forecast waiting times under baseline routing. The scheduler then re-routes flexible EVs away from the predicted bottleneck:

```
predicted arrivals → DT what-if simulation → predicted waiting times → scheduler inverse-weight routing → load-balanced assignment
```

This creates a **closed feedback loop**: if Caltech is predicted to be congested, more flexible EVs are sent to JPL, smoothing load in real time. The baseline, relying on a static historical average, cannot react to transient spikes. Metrics like `avg_waiting_time`, `avg_queue_length`, and `load_imbalance` are compared across both modes in `evaluation.py`.

## Data Flow

```
Training:  ground_truth → edge nodes (local train) → FedAvg every 24h → global model
Evaluation: ground_truth (demand) → edge predictions → DT forecast → scheduler route → queue simulator → metrics
```

## Output Files

Three CSV files are written to the project root by `main.py`.

### results_dt.csv / results_baseline.csv

One row per simulated hour (26,515 rows for the full dataset). Both have identical columns; only `routed_to_*` and `sim_*` columns differ, reflecting the two scheduling policies.

| Column | Description |
|--------|-------------|
| `timestamp` | Simulation hour |
| `t` | Timestep index |
| `total_demand` | Total EV arrivals this hour (ground truth Caltech + JPL) |
| `ground_truth_caltech_arrivals` | Actual historical arrivals at Caltech |
| `ground_truth_jpl_arrivals` | Actual historical arrivals at JPL |
| `caltech_predicted_arrivals` | Edge node demand forecast for next hour at Caltech |
| `jpl_predicted_arrivals` | Edge node demand forecast for next hour at JPL |
| `caltech_dt_predicted_queue` | DT-predicted queue length at Caltech |
| `jpl_dt_predicted_queue` | DT-predicted queue length at JPL |
| `caltech_dt_predicted_wait` | DT-predicted waiting time (minutes) at Caltech |
| `jpl_dt_predicted_wait` | DT-predicted waiting time (minutes) at JPL |
| `caltech_dt_predicted_util` | DT-predicted charger utilization at Caltech |
| `jpl_dt_predicted_util` | DT-predicted charger utilization at JPL |
| `caltech_congestion` | DT congestion score (0–1 composite) for Caltech |
| `jpl_congestion` | DT congestion score (0–1 composite) for JPL |
| `routed_to_caltech` | EVs routed to Caltech (fixed proportion in baseline, DT-guided in DT run) |
| `routed_to_jpl` | EVs routed to JPL |
| `sim_caltech_queue` | Actual queue after routing (from queue simulator) |
| `sim_jpl_queue` | Actual queue after routing (from queue simulator) |
| `sim_caltech_wait` | Actual waiting time (minutes) after routing |
| `sim_jpl_wait` | Actual waiting time (minutes) after routing |
| `sim_caltech_util` | Actual charger utilization after routing |
| `sim_jpl_util` | Actual charger utilization after routing |

**Key difference:** `results_dt.csv` uses DT-guided inverse-congestion allocation; `results_baseline.csv` uses a fixed historical demand split (~40.4% Caltech, 59.6% JPL).

### evaluation_summary.csv

One row per metric comparing DT-assisted vs Baseline performance across prediction accuracy, DT prediction error, and scheduling outcomes (average wait, queue, utilization balance, peak congestion).

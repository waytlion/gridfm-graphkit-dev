# Forecast OPF Implementation Summary
## What 
A one-step-ahead forecasting task that predicts the complete optimal power flow state at time *t+1* given the state at time *t*.

### Key Difference from Standard OPF

- **Standard OPF:**  
  Masks certain features (e.g., voltages) → predicts only masked features  

- **Forecast OPF:**  
  Sees all features at time *t* → predicts dynamic features at time *t+1*  

---

## Why
Enables joint load forecasting + optimal dispatch in a single model:
- Predict how demand changes over time  
- Predict optimal generator response to forecasted loads  
- No separate load forecasting step required  
---

## Implementation Approach

### 1. Dataset (`HeteroGridForecastDatasetDisk`)

- Loads pairs: state at *t* (input) and state at *t+1* (target)  
- Filters targets to only:
  - **Bus (5 features):** `[Pd, Qd, Qg, Vm, Va]` (indices 0–4)  
  - **Generator (1 feature):** `[Pg]` (index 0)  

### 2. Masking (`AddOPFForecastingMask`)

- Marks the 5 dynamic bus features as "to be predicted"  
- Model still sees all 15 input features  

### 3. Task (`ForecastOPFTask`)

Inherits from `OptimalPowerFlowTask`, overrides:
  - training_step()
    - logs individual loss components (Forecast bus MSE, Forecast gen MSE, physics)
  - test_step()
    - adds MAE metrics for all 6 predicted features + converts to OPF format for physics evaluation
  - on_test_end()
    - generates forecast MAE CSV, then delegates to parent for RMSE/plots

### 4. Model (`GNS_heterogeneous`)

Modified to support `task_name == "ForecastOPF"`:

- Decoder outputs:
  - 5 bus features (instead of 2 for OPF)  
  - 1 generator feature  
- Physics decoder registered for ForecastOPF task  

### 5. Loss Function (`loss.py`)
- Added ForecastLossMSE for bus and gen

---

## Usage
```bash
gridfm_graphkit train    --config examples/config/forecasting_test.yaml --data_path data/
gridfm_graphkit evaluate --config examples/config/forecasting_test.yaml --data_path data/ --model_path <path_to_model>
```

---

## Files Changed (for review)

### New Files (core implementation)

| File | Description |
|------|-------------|
| `gridfm_graphkit/tasks/forecast_opf_task.py` | ForecastOPF task — test metrics, MAE logging, format conversion |
| `gridfm_graphkit/datasets/powergrid_hetero_forecast_dataset.py` | Forecast dataset — loads (t, t+1) pairs |
| `gridfm_graphkit/datasets/hetero_powergrid_forecast_datamodule.py` | DataModule for forecast — temporal split, pairs handling |
| `examples/config/forecasting_test.yaml` | Quick-test config (1 epoch, CPU, case14) |
| `examples/config/HGNS_ForecastingOPF_case14.yaml` | Full training config for case14 |

### Modified Files (key changes)

| File | What changed |
|------|-------------|
| `gridfm_graphkit/tasks/opf_task.py` | Extracted `_compute_opf_metrics()` so ForecastOPF can reuse physics metrics |
| `gridfm_graphkit/datasets/powergrid_hetero_dataset.py` | Feature lists extracted as class constants (`BUS_FEATURES`, etc.) |
| `gridfm_graphkit/training/loss.py` | Added `ForecastBusMSE` and `ForecastGenMSE` loss functions |
| `gridfm_graphkit/models/gnn_heterogeneous_gns.py` | Decoder outputs 5 bus features for ForecastOPF; physics decoder registered |
| `gridfm_graphkit/datasets/masking.py` | Added `AddOPFForecastingMask` |
| `gridfm_graphkit/datasets/task_transforms.py` | Registered forecast mask transform |
| `gridfm_graphkit/cli.py` | Routes `ForecastOPF` to forecast DataModule |
| `gridfm_graphkit/datasets/utils.py` | Temporal split logic for chronological train/val/test |
| `gridfm_graphkit/tasks/__init__.py` | Registered `ForecastOPFTask` import |

### Other Modified (minor / infra)
- `gridfm_graphkit/__main__.py` — CLI subcommand setup
- `gridfm_graphkit/datasets/normalizers.py` — minor adjustments
- `gridfm_graphkit/datasets/hetero_powergrid_datamodule.py` — split logging
- `gridfm_graphkit/tasks/base_task.py`, `pf_task.py`, `se_task.py`, `utils.py` — minor updates
- `gridfm_graphkit/models/utils.py` — minor
- `tests/test_pipeline.py`, `tests/config/datamodule_test_base_config3.yaml` — test updates

### Recommended Review Order
1. **Start with** `examples/config/forecasting_test.yaml` — understand the task config
2. **Data layer**: `powergrid_hetero_forecast_dataset.py` → `hetero_powergrid_forecast_datamodule.py`
3. **Task layer**: `opf_task.py` (see `_compute_opf_metrics`) → `forecast_opf_task.py`
4. **Model**: `gnn_heterogeneous_gns.py` (search for "ForecastOPF")
5. **Loss**: `loss.py` (search for "Forecast")

### How to Run
```bash
# Quick test (1 epoch, CPU, case14, ~2 min)
gridfm_graphkit train --config examples/config/forecasting_test.yaml --data_path data/ --exp_name "review" --run_name "quick_test"

# View results
mlflow ui --backend-store-uri mlruns
# → open http://localhost:5000
```
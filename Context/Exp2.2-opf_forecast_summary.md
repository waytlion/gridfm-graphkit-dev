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

---

### 2. Masking (`AddOPFForecastingMask`)

- Marks the 5 dynamic bus features as "to be predicted"  
- Model still sees all 15 input features  
- Enables selective loss computation on predicted features only  

---

### 3. Task (`ForecastOPFTask`)

Inherits from `OptimalPowerFlowTask`, overrides:

- `shared_step()`  
  - Computes MSE on predicted features  
  - Adds physics loss  

- `test_step()`  
  - Adds MAE metrics for load forecasting quality  

---

### 4. Model (`GNS_heterogeneous`)

Modified to support `task_name == "ForecastOPF"`:

- Decoder outputs:
  - 5 bus features (instead of 2 for OPF)  
  - 1 generator feature  
- Physics decoder registered for ForecastOPF task  

---

### 5. Loss Function

**Weights:** `[0.1, 0.1, 0.8]`  
(Physics-heavy to ensure feasibility)

---



## Usage
python -m gridfm_graphkit train  --config .\examples\config\forecasting_test.yaml --data_path data/
python -m gridfm_graphkit evaluate  --config .\examples\config\forecasting_test.yaml --data_path data/
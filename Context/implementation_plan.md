# Forecast OPF Task - Implementation Plan

**Date:** February 11, 2026  
**Status:** Planning Phase  
**Purpose:** Enable one-step-ahead forecasting of optimal power flow states (t → t+1)

---

## Table of Contents 
1. [High-Level Goal](#1-high-level-goal)
2. [Abstract Requirements](#2-abstract-requirements)
3. [Design Decisions & Reasoning](#3-design-decisions--reasoning)
4. [Code Implementation Plan](#4-code-implementation-plan)
5. [Testing Strategy](#5-testing-strategy)
6. [Future Considerations](#6-future-considerations)

---

## 0. Questions:
- model: Should rlly predict Qg when it gets completly overwritten by physics decoder? 
## 1. High-Level Goal

### What We're Building
A new task (`ForecastOPFTask`) that predicts the complete optimal power flow state at time t+1, given the complete state at time t.

**Key difference from existing OPF task:**
- **Existing OPF:** Masks certain features (voltages, generation) → predicts masked features given loads
- **Forecast OPF:** NO masking → predicts ALL features (loads, voltages, generation) at t+1

### Why It's Needed
1. **Load Forecasting:** Predict how demand changes over time
2. **Optimal Dispatch:** Predict how generators should respond to forecasted loads
3. **Joint Learning:** Learn temporal dynamics + OPF optimization in one model
4. **End-to-End:** Direct t→t+1 prediction without separate load forecasting step

### Expected Behavior
```
Input (t):  Complete OPF state
            - Loads: [P_d, Q_d]
            - Voltages: [V_m, V_a]  
            - Generation: [P_g]
            - Static features: [limits, costs, bus types]
            ↓
Model:      GNN learns temporal patterns + OPF principles
            ↓
Output (t+1): 
| Feature | Location | Predicted? | Physics-Corrected? | Why                           |
|----------|----------|------------|--------------------|------------------------------|
| Pd       | Bus      | Yes        | No                 | Exogenous load                |
| Qd       | Bus      | Yes        | No                 | Exogenous load                |
| Qg       | Bus      | Initially  | Yes (replaced)     | Derived from power balance    |
| Vm       | Bus      | Yes        | Optional           | Can be physics-corrected      |
| Va       | Bus      | Yes        | Optional           | Can be physics-corrected      |
| Pg       | Gen      | Yes        | No                 | Dispatch decision             |
```


Gen (Pg): 
**3 Objectives:**
- Accurate load forecasts (low MAE on P_d, Q_d)
- Physically feasible predictions (minimize physical error term)
- optimal dispatch (low gen cost)

---

## 2. Abstract Requirements

### 2.1 Data Flow
- **Input:** Use `LitGridHeteroForecastDataModule` (already exists)
- **Transform:** New `ForecastOPFTransform` (no masking, full masks)
- **Output:** Predict all dynamic features at t+1

### 2.2 Feature Dimensions

**Bus Features:**
```
Input:  [num_bus, 15]  # All features at time t
Output: [num_bus, 5]   # Dynamic features: [P_d, Q_d, Q_g, V_m, V_a]
                       # Indices: [0, 1, 2, 3, 4]
Static features (not predicted): bus types, limits, shunt, nominal voltage
```

**Generator Features:**
```
Input:  [num_gen, 6]   # All features at time t
Output: [num_gen, 1]   # Active power: [P_g]
                       # Index: [0]
Static features (not predicted): limits [P_min, P_max], costs [C0, C1, C2]
```

### 2.3 Masking Strategy

**Full Masks (all True):**
```python
mask_dict = {
    'bus': torch.ones([num_bus, 15], dtype=bool),   # All True
    'gen': torch.ones([num_gen, 6], dtype=bool),    # All True
    'PQ': bus.x[:, PQ_H] == 1,   # Bus type flags (unchanged)
    'PV': bus.x[:, PV_H] == 1,
    'REF': bus.x[:, REF_H] == 1,
}
```

**Why full masks?**
- Model uses masks to select predicted vs. ground truth values
- `mask=True` → use model prediction
- `mask=False` → use ground truth (bypass model)
- For forecast: we want ALL predictions used, so all masks = True

### 2.4 Loss Function Design

**Components:**
```python
total_loss = w_bus * MSE(bus_features) + 
             w_gen * MSE(gen_features) + 
             w_physics * LayeredWeightedPhysics
```

**Initial weights:** Equal balance
```yaml
loss_weights:
  bus: 0.33      # Loads + voltages
  gen: 0.33      # Generation
  physics: 0.33  # Power balance
```

**Loss Details:**

1. **Bus MSE:** All 5 dynamic features [P_d, Q_d, Q_g, V_m, V_a]
2. **Gen MSE:** Active power [P_g]
3. **Physics:** Reuse existing `LayeredWeightedPhysics`
   - Active power balance: sum(P_g) - sum(P_d) - P_losses = 0
   - Reactive power balance: sum(Q_g) - sum(Q_d) - Q_losses = 0
   - Layered penalties (early GNN layers penalized more)

### 2.5 Model Architecture

**Use existing:** `GNS_heterogeneous` (no changes needed)

**Configuration:**
```yaml
model:
  type: GNS_heterogeneous
  hidden_size: 48
  num_layers: 12
  attention_head: 8
  
  input_bus_dim: 15      # All features at t
  output_bus_dim: 5      # Dynamic features to predict
  
  input_gen_dim: 6       # All features at t
  output_gen_dim: 1      # P_g to predict
```

**Keep sigmoid bounds:** Ensures predictions respect physical limits (V ∈ [0.9, 1.1], P_g ∈ [P_min, P_max])

---

## 4. Code Implementation Plan

### 4.1 Files to CREATE

#### **File 1: `gridfm_graphkit/datasets/task_transforms.py` (add to existing)**

**Add new transform class:**

```python
@TRANSFORM_REGISTRY.register("ForecastOPF")
class ForecastOPFTransforms(Compose):
    """
    Transform for forecast OPF task - no feature masking.
    All features visible at time t, all features predicted at time t+1.
    """
    
    def __init__(self, args):
        transforms = [
            RemoveInactiveBranches(),      # Reuse existing
            RemoveInactiveGenerators(),    # Reuse existing
            CreateFullMasks(),             # NEW - create all-True masks
        ]
        super().__init__(transforms)
```

**Purpose:** Prepare data for forecast task without masking

**Key difference from OPF transforms:**
- No `AddOPFHeteroMask()` (which masks voltages/generation)
- No `ApplyMasking()` (which zeros out masked features)
- Instead: `CreateFullMasks()` for model compatibility

---

#### **File 2: `gridfm_graphkit/datasets/masking.py` (add to existing)**

**Add new masking class:**

```python
class CreateFullMasks(BaseTransform):
    """
    Creates full masks (all True) for forecast task.
    Model will use all predictions, no ground truth passthrough.
    """
    
    def forward(self, data: HeteroData) -> HeteroData:
        # Get feature tensors
        bus_x = data.x_dict["bus"]
        gen_x = data.x_dict["gen"]
        branch_attr = data.edge_attr_dict[("bus", "connects", "bus")]
        
        # Create all-True masks (everything is predicted)
        mask_bus = torch.ones_like(bus_x, dtype=torch.bool)
        mask_gen = torch.ones_like(gen_x, dtype=torch.bool)
        mask_branch = torch.ones_like(branch_attr, dtype=torch.bool)
        
        # Bus type flags (still needed for physics decoder)
        mask_PQ = bus_x[:, PQ_H] == 1
        mask_PV = bus_x[:, PV_H] == 1
        mask_REF = bus_x[:, REF_H] == 1
        
        # Store masks
        data.mask_dict = {
            "bus": mask_bus,
            "gen": mask_gen,
            "branch": mask_branch,
            "PQ": mask_PQ,
            "PV": mask_PV,
            "REF": mask_REF,
        }
        
        return data
```

**Purpose:** Create masks that tell model to use all predictions

**Why needed:** Model expects `mask_dict` in forward pass, uses it to decide predicted vs. ground truth values

---

#### **File 3: `gridfm_graphkit/tasks/forecast_opf_task.py` (NEW FILE)**

**Create new task:**

```python
from gridfm_graphkit.tasks.base_task import BaseTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.training.losses import LayeredWeightedPhysics
import torch.nn.functional as F

@TASK_REGISTRY.register("ForecastOPF")
class ForecastOPFTask(BaseTask):
    """
    Forecast OPF Task: Predict complete state at t+1 given state at t.
    
    No masking - predicts ALL features:
    - Load forecasting: P_d, Q_d
    - Voltage solution: V_m, V_a
    - Optimal dispatch: P_g
    """
    
    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        
        # Initialize model (copy from OptimalPowerFlowTask)
        from gridfm_graphkit.io.param_handler import load_model
        self.net = load_model(args, data_normalizers)
        
        # Loss functions
        self.physics_loss = LayeredWeightedPhysics(
            base_weight=args.training.loss_args[2].get('base_weight', 0.5)
        )
        
        # Loss weights
        self.loss_weights = {
            'bus': args.training.loss_weights[0],      # 0.33
            'gen': args.training.loss_weights[1],      # 0.33
            'physics': args.training.loss_weights[2],  # 0.33
        }
    
    def forward(self, batch):
        """Forward pass through model"""
        return self.net(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict,
            mask_dict=batch.mask_dict,
        )
    
    def shared_step(self, batch, batch_idx, stage):
        """
        Shared step for train/val/test.
        Computes forward pass and all losses.
        """
        # Forward pass
        output = self.forward(batch)
        
        # Compute feature MSE losses
        loss_bus = F.mse_loss(output['bus'], batch.y_dict['bus'])
        loss_gen = F.mse_loss(output['gen'], batch.y_dict['gen'])
        
        # Compute physics loss (power balance)
        loss_physics = self.physics_loss(output, batch.y_dict, batch)
        
        # Weighted total loss
        total_loss = (
            self.loss_weights['bus'] * loss_bus +
            self.loss_weights['gen'] * loss_gen +
            self.loss_weights['physics'] * loss_physics
        )
        
        # Log individual losses
        self.log(f"{stage}/loss_bus", loss_bus)
        self.log(f"{stage}/loss_gen", loss_gen)
        self.log(f"{stage}/loss_physics", loss_physics)
        self.log(f"{stage}/loss_total", total_loss)
        
        # Compute additional metrics
        if stage in ['val', 'test']:
            metrics = self.compute_metrics(output, batch)
            for key, value in metrics.items():
                self.log(f"{stage}/{key}", value)
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='val')
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage='test')
        
        # Store predictions for end-of-epoch visualization
        self.test_predictions.append(self.detach_output(output))
        self.test_targets.append(self.detach_batch(batch))
        
        return loss
    
    def compute_metrics(self, output, batch):
        """
        Compute detailed metrics for validation/testing.
        
        Returns dict with:
        - MAE per feature type (load, voltage, generation)
        - Power imbalance
        - Generation cost gap
        """
        metrics = {}
        
        # Denormalize for physical units
        output_denorm = self.denormalize(output)
        target_denorm = self.denormalize(batch.y_dict)
        
        # Bus metrics (load + voltage)
        bus_pred = output_denorm['bus']
        bus_target = target_denorm['bus']
        
        # Load forecasting accuracy
        metrics['mae_P_d'] = (bus_pred[:, 0] - bus_target[:, 0]).abs().mean()
        metrics['mae_Q_d'] = (bus_pred[:, 1] - bus_target[:, 1]).abs().mean()
        
        # Voltage prediction accuracy
        metrics['mae_V_m'] = (bus_pred[:, 3] - bus_target[:, 3]).abs().mean()
        metrics['mae_V_a'] = (bus_pred[:, 4] - bus_target[:, 4]).abs().mean()
        
        # Generation metrics
        gen_pred = output_denorm['gen']
        gen_target = target_denorm['gen']
        metrics['mae_P_g'] = (gen_pred[:, 0] - gen_target[:, 0]).abs().mean()
        
        # Power balance check
        P_gen_total = gen_pred[:, 0].sum()
        P_load_total = bus_pred[:, 0].sum()
        metrics['power_imbalance'] = (P_gen_total - P_load_total).abs()
        
        # Economic cost (for monitoring)
        # Assumes cost coefficients available in batch.x_dict['gen']
        gen_x = batch.x_dict['gen']
        C0 = gen_x[:, C0_H]
        C1 = gen_x[:, C1_H]
        C2 = gen_x[:, C2_H]
        
        cost_pred = (C0 + C1 * gen_pred[:, 0] + C2 * gen_pred[:, 0]**2).sum()
        cost_target = (C0 + C1 * gen_target[:, 0] + C2 * gen_target[:, 0]**2).sum()
        metrics['cost_gap'] = (cost_pred - cost_target).abs()
        metrics['cost_gap_pct'] = ((cost_pred - cost_target) / cost_target * 100).abs()
        
        return metrics
    
    def on_test_epoch_end(self):
        """
        Called at end of test epoch.
        Generate visualizations and aggregate metrics.
        """
        # Aggregate all test predictions
        all_outputs = self.aggregate_predictions(self.test_predictions)
        all_targets = self.aggregate_predictions(self.test_targets)
        
        # Create plots
        self.plot_load_forecasts(all_outputs, all_targets)
        self.plot_voltage_predictions(all_outputs, all_targets)
        self.plot_generation_dispatch(all_outputs, all_targets)
        self.plot_power_balance(all_outputs, all_targets)
        
        # Log to experiment tracker
        if self.logger:
            self.logger.log_plot("test/load_forecast", self.fig_loads)
            self.logger.log_plot("test/voltage_prediction", self.fig_voltages)
            self.logger.log_plot("test/generation_dispatch", self.fig_generation)
        
        # Clear stored predictions
        self.test_predictions.clear()
        self.test_targets.clear()
    
    def plot_load_forecasts(self, outputs, targets):
        """Scatter plot: predicted vs. actual loads"""
        # Implementation similar to OptimalPowerFlowTask plots
        pass
    
    def plot_voltage_predictions(self, outputs, targets):
        """Scatter plot: predicted vs. actual voltages"""
        pass
    
    def plot_generation_dispatch(self, outputs, targets):
        """Scatter plot: predicted vs. actual generation"""
        pass
    
    def plot_power_balance(self, outputs, targets):
        """Histogram of power imbalance violations"""
        pass
```

**Purpose:** Main task implementation for forecast OPF

**Key methods:**
- `forward()`: Pass data through model
- `shared_step()`: Compute losses (train/val/test)
- `compute_metrics()`: Detailed performance metrics
- `on_test_epoch_end()`: Visualizations
- Plot methods: Analysis of predictions

---

### 4.2 Files to MODIFY

#### **File 1: `gridfm_graphkit/io/param_handler.py`**

**Modification:** Register new transform in `get_task_transforms()`

```python
def get_task_transforms(args):
    """Get transforms based on task name"""
    
    task_name = args.task.task_name
    
    # Existing transforms
    if task_name == "OptimalPowerFlow":
        return TRANSFORM_REGISTRY["OptimalPowerFlow"](args)
    elif task_name == "PowerFlow":
        return TRANSFORM_REGISTRY["PowerFlow"](args)
    elif task_name == "ForecastOPF":  # NEW
        return TRANSFORM_REGISTRY["ForecastOPF"](args)
    # ... other tasks
```

**Purpose:** Route to correct transform based on task name

---

#### **File 2: `gridfm_graphkit/cli.py`**

**Modification:** Use forecast datamodule when appropriate

```python
def main_cli(args):
    # Load config
    config_args = NestedNamespace(**yaml.safe_load(args.config))
    
    # Check if forecast task
    if config_args.task.task_name in ["ForecastOPF"]:  # Can add more forecast tasks
        from gridfm_graphkit.datasets.hetero_powergrid_forecast_datamodule import LitGridHeteroForecastDataModule
        litGrid = LitGridHeteroForecastDataModule(config_args, args.data_path)
    else:
        litGrid = LitGridHeteroDataModule(config_args, args.data_path)
    
    # Rest of pipeline unchanged
    model = get_task(config_args, litGrid.data_normalizers)
    trainer = L.Trainer(...)
    trainer.fit(model=model, datamodule=litGrid)
```

**Purpose:** Automatically select forecast datamodule for forecast tasks

---

#### **File 3: Create new config file**

**File:** `examples/config/HGNS_ForecastOPF_case14.yaml`

```yaml
task:
  task_name: ForecastOPF  # NEW task

data:
  baseMVA: 100
  mask_value: 0.0  # Not used (no masking) but kept for compatibility
  normalization: HeteroDataMVANormalizer
  networks:
    - case14_ieee
  scenarios:
    - 250000
  test_ratio: 0.1
  val_ratio: 0.1
  workers: 32
  split_by_load_scenario_idx: true

model:
  attention_head: 8
  edge_dim: 10
  hidden_size: 48
  input_bus_dim: 15     # All features at t
  output_bus_dim: 5     # P_d, Q_d, Q_g, V_m, V_a (changed from 2)
  input_gen_dim: 6      # All features at t
  output_gen_dim: 1     # P_g (unchanged)
  num_layers: 12
  type: GNS_heterogeneous

optimizer:
  beta1: 0.9
  beta2: 0.999
  learning_rate: 0.0005
  lr_decay: 0.7
  lr_patience: 5

training:
  batch_size: 64
  epochs: 200
  loss_weights:
    - 0.33  # bus (loads + voltages)
    - 0.33  # gen (P_g)
    - 0.33  # physics (power balance)
  losses:
    - BusMSE
    - GenMSE
    - LayeredWeightedPhysics
  loss_args:
    - {}
    - {}
    - base_weight: 0.5  # For layered physics
  accelerator: auto
  devices: auto
  strategy: auto

seed: 0
verbose: true

callbacks:
  patience: 100
  tol: 0
```

**Purpose:** Configuration for forecast OPF experiments

**Key changes from OPF config:**
- `task_name: ForecastOPF`
- `output_bus_dim: 5` (was 2)
- Loss weights adjusted for 3 components

---

### 4.3 Component Interactions

```
Config YAML (ForecastOPF)
    ↓
CLI detects "ForecastOPF" task
    ↓
LitGridHeteroForecastDataModule loaded
    ├→ Loads t and t+1 pairs (HeteroGridForecastDatasetDisk)
    ├→ Applies ForecastOPFTransform
    │   ├→ RemoveInactiveBranches
    │   ├→ RemoveInactiveGenerators
    │   └→ CreateFullMasks (all True)
    └→ Splits train/val/test
    ↓
ForecastOPFTask initialized
    ├→ Loads GNS_heterogeneous model
    ├→ Sets up losses: [BusMSE, GenMSE, LayeredWeightedPhysics]
    └→ Configures metrics & visualization
    ↓
Training Loop
    ├→ training_step: forward → compute losses → backprop
    ├→ validation_step: forward → compute losses + metrics
    └→ test_step: forward → losses + metrics + store for viz
    ↓
Test Epoch End
    └→ Generate plots: load forecasts, voltages, dispatch, power balance
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**Test 1: Transform Output**
```python
def test_forecast_opf_transform():
    """Verify ForecastOPFTransform creates full masks"""
    transform = ForecastOPFTransforms(args)
    data = create_sample_hetero_data()
    
    transformed = transform(data)
    
    # Check masks exist
    assert 'mask_dict' in transformed
    
    # Check all masks are True
    assert transformed.mask_dict['bus'].all()
    assert transformed.mask_dict['gen'].all()
    
    # Check bus types preserved
    assert (transformed.mask_dict['PQ'].sum() + 
            transformed.mask_dict['PV'].sum() + 
            transformed.mask_dict['REF'].sum()) == num_buses
```

**Test 2: Loss Computation**
```python
def test_forecast_task_losses():
    """Verify loss computation is correct"""
    task = ForecastOPFTask(args, normalizers)
    batch = create_sample_batch()
    
    output = task.forward(batch)
    loss = task.shared_step(batch, 0, 'train')
    
    # Check loss is scalar
    assert loss.dim() == 0
    
    # Check loss is positive
    assert loss > 0
    
    # Check individual losses logged
    assert 'train/loss_bus' in task.logged_metrics
    assert 'train/loss_gen' in task.logged_metrics
    assert 'train/loss_physics' in task.logged_metrics
```

**Test 3: Metrics Computation**
```python
def test_forecast_metrics():
    """Verify metrics are computed correctly"""
    task = ForecastOPFTask(args, normalizers)
    batch = create_sample_batch()
    output = task.forward(batch)
    
    metrics = task.compute_metrics(output, batch)
    
    # Check expected metrics exist
    assert 'mae_P_d' in metrics
    assert 'mae_Q_d' in metrics
    assert 'mae_V_m' in metrics
    assert 'mae_P_g' in metrics
    assert 'power_imbalance' in metrics
    assert 'cost_gap' in metrics
```

### 5.2 Integration Tests

**Test 1: End-to-End Pipeline**
```python
def test_forecast_opf_pipeline():
    """Test complete training pipeline"""
    # Load config
    args = load_config("examples/config/HGNS_ForecastOPF_case14.yaml")
    
    # Initialize datamodule
    datamodule = LitGridHeteroForecastDataModule(args, data_path)
    datamodule.setup('fit')
    
    # Initialize task
    task = ForecastOPFTask(args, datamodule.data_normalizers)
    
    # Create trainer (fast_dev_run)
    trainer = L.Trainer(fast_dev_run=True)
    
    # Run training
    trainer.fit(task, datamodule)
    
    # Check training completed without errors
    assert trainer.current_epoch > 0
```

**Test 2: Datamodule Compatibility**
```python
def test_forecast_datamodule_output():
    """Verify datamodule outputs correct structure"""
    datamodule = LitGridHeteroForecastDataModule(args, data_path)
    datamodule.setup('fit')
    
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    # Check batch structure
    assert 'x_dict' in batch
    assert 'y_dict' in batch
    assert 'mask_dict' in batch
    
    # Check mask values
    assert batch.mask_dict['bus'].all()  # All True
    assert batch.mask_dict['gen'].all()
```

### 5.3 Validation Tests

**Test 1: Physical Feasibility**
```python
def test_power_balance():
    """Verify predictions satisfy power balance"""
    task = trained_forecast_task
    test_loader = datamodule.test_dataloader()
    
    imbalances = []
    for batch in test_loader:
        output = task.forward(batch)
        
        P_gen = output['gen'][:, 0].sum()
        P_load = output['bus'][:, 0].sum()
        imbalance = (P_gen - P_load).abs()
        
        imbalances.append(imbalance.item())
    
    # Check 95% of predictions have small imbalance
    assert np.percentile(imbalances, 95) < 0.01  # 1% of baseMVA
```

**Test 2: Load Forecast Accuracy**
```python
def test_load_forecast_mae():
    """Verify load forecasts meet accuracy threshold"""
    task = trained_forecast_task
    mae_loads = evaluate_load_mae(task, test_loader)
    
    # Example thresholds (adjust based on data characteristics)
    assert mae_loads['P_d'] < 5.0  # MW
    assert mae_loads['Q_d'] < 2.0  # Mvar
```

### 5.4 Comparison Tests

**Test: Compare to Baseline**
```python
def test_vs_persistence_model():
    """Compare forecast to simple persistence baseline (t+1 = t)"""
    task = trained_forecast_task
    
    # Forecast model MAE
    mae_forecast = evaluate_mae(task, test_loader)
    
    # Persistence baseline MAE (predict t+1 = t)
    mae_persistence = evaluate_persistence_mae(test_loader)
    
    # Forecast should beat persistence
    assert mae_forecast < mae_persistence
```

---

## 6. Future Considerations

### 6.1 Potential Enhancements

**Input History window (t-n)**
- concat to one graph -> feed in to predict t+1

**Loss Granularity:**
- Separate weights for load, voltage, generation
- Allows emphasizing load forecasting vs. dispatch optimization
- Implementation: Split bus features by index

**Economic Cost in Loss:**
- Add small penalty for generation cost deviation
- Helps generalization to unseen load patterns
- Formula: `loss += λ * |predicted_cost - target_cost|`


**Multi-step Forecasting:**
- Predict multiple timesteps: t+1, t+2, ..., t+k
- Requires model architecture changes (recurrent or autoregressive)
- Loss over all timesteps

### 6.2 Monitoring During Training

**Key metrics to watch:**

1. **Loss components balance:**
   - If `loss_physics >> loss_features`: Model prioritizing feasibility over accuracy
   - If `loss_features >> loss_physics`: Model accurate but not physically consistent
   - Adjust weights if imbalanced

2. **Load forecast accuracy:**
   - Track MAE on P_d, Q_d separately
   - If poor: Consider increasing `w_bus` or adding load-specific loss

3. **Power imbalance:**
   - Should decrease during training
   - If stagnant: Physics loss weight too low

4. **Economic cost gap:**
   - Monitor predicted vs. target cost
   - Large gaps on test set → consider adding cost to loss

### 6.3 Experiment Variations

**Ablation studies to consider:**

1. **Loss weights:**
   - Try different ratios: [0.2, 0.2, 0.6] (physics-heavy)
   - Try [0.4, 0.4, 0.2] (accuracy-heavy)

2. **Model capacity:**
   - Vary `hidden_size`: 32, 48, 64, 96
   - Vary `num_layers`: 6, 12, 18

3. **Data augmentation:**
   - Add noise to inputs (robustness test)
   - Test on out-of-distribution loads

4. **Architecture variants:**
   - Try GraphTransformer instead of GAT
   - Compare to simpler baseline (MLP on concatenated features)

### 6.4 Deployment Considerations

**For real-world use:**

1. **Inference speed:**
   - Profile forward pass time
   - Consider model compression if needed

2. **Safety checks:**
   - Post-process predictions to enforce hard constraints
   - Fallback to OPF solver if predictions infeasible

3. **Uncertainty handling:**
   - Don't trust predictions outside training distribution
   - Monitor input distribution shift

4. **Retraining strategy:**
   - Periodically retrain on new data
   - Online learning / fine-tuning

---

## Appendix: Quick Reference
### Feature Indices (from globals.py)

**End of Implementation Plan**
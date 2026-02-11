# Architecture
┌─────────────────────────────────────────────────────────────┐
│                     CLI Entry Point                          │
│              (gridfm_graphkit/__main__.py)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Main Pipeline (cli.py)                     │
│  - Loads config YAML                                         │
│  - Initializes DataModule                                    │
│  - Creates Task (model + training logic)                     │
│  - Runs PyTorch Lightning Trainer                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌────────┐   ┌─────────┐   ┌─────────┐
    │  DATA  │   │  MODEL  │   │  TASK   │
    └────────┘   └─────────┘   └─────────┘




# Execution Flow
Config YAML
    ↓
[CLI Parser] → Load config as NestedNamespace
    ↓
[DataModule.setup()]
    ├→ Load HeteroGridDatasetDisk (case14_ieee, 250k scenarios)
    ├→ Apply HeteroDataMVANormalizer
    ├→ Apply OptimalPowerFlowTransforms (mask P_g)
    └→ Split 200k/25k/25k (train/val/test)
    ↓
[get_task()] → OptimalPowerFlowTask
    ├→ load_model() → GNS_heterogeneous (12 layers, 48 hidden)
    └→ get_loss_functions() → [Physics(0.1), GenMSE(0.1), BusMSE(0.8)]
    ↓
[Trainer.fit()]
    └→ For 200 epochs:
        ├→ training_step(): Forward → Compute losses → Backprop
        └→ validation_step(): Evaluate on val set
    ↓
[Trainer.test()]
    └→ test_step(): Metrics + Visualizations


    
# Data flow
Raw Data (Disk)
    │
    ▼
┌──────────────────────────────────────────────┐
│  HeteroGridDatasetDisk                       │
│  - Loads graph data from disk                │
│  - Applies normalization                     │
│  - Returns HeteroData objects                │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  LitGridHeteroDataModule                     │
│  - Splits into train/val/test                │
│  - Creates DataLoaders                       │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Task (e.g., PowerFlowTask)                  │
│  - training_step: Forward pass + loss        │
│  - validation_step: Metrics                  │
│  - test_step: Detailed evaluation            │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Model (e.g., GNS_heterogeneous)             │
│  - GNN forward pass                          │
│  - Physics decoder (optional)                │
└──────────────────────────────────────────────┘


# important files:
-  gridfm_graphkit\datasets\globals.py: Clarifies Feature Indices
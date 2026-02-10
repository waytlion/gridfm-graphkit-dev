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


    
# CLAUDE.md — gridfm-graphkit-dev

## Purpose
Fork of a colleague's graphkit GNN library (changes merge back upstream). Holds the two ML arms of the
thesis: **surrogate** (forecast → GNN replaces the OPF solver) and **E2E** (ST-GNN, load history →
forecast + dispatch). Thesis context + global rules: `~/CLAUDE.md`. Results/rationale: `docs/findings.md`.

## Map
```
gridfm_graphkit/
  cli.py                       entry: python -m gridfm_graphkit {train,evaluate}; InferenceTimer callback
  tasks/
    base_task.py               optimizer (AdamW) + ReduceLROnPlateau scheduler
    opf_task.py                base OPF task: test_step → _override_residual_load → _compute_opf_metrics → _after_opf_metrics
    opf_twostep_task.py        surrogate arm; accumulates violations + system totals; writes canonical metrics.csv
    st_forecast_opf_task.py    E2E arm; denorm + forecast collection; writes forecast.csv + metrics.csv
    constraint_metrics.py      canonical metric core (shared by both arms)
    perbus_residual_dump.py    PerBusResidualAccumulator (composition): per-bus signed residuals + state dumps
    reconstruction_tasks.py    shared_step; forward timing gated by _time_forward
    forecast_opf_task.py       colleague's single-step forecast task (unused in thesis)
  models/
    gnn_heterogeneous_gns.py   surrogate GNN (GNS_heterogeneous, 12-layer HeteroConv(TransformerConv))
    st_gnn_heterogeneous.py    E2E ST-GNN: UnmaskedSpatialEncoder + temporal encoder (TCN/Transformer) + heads
    temporal_encoders.py, utils.py (PhysicsDecoderOPF etc.)
  training/loss.py             ST_ForecastBusMSE, ST_ForecastGenMSE, ST_PhysicsForecastLoss, MixedLoss
  datasets/globals.py          feature index constants (PD_H, QD_H, VM_H, VA_H, PG_H, QG_H, …)
  datasets/*temporal*          E2E dataset/datamodule (cyclical time features, temporal_split, train_window)
examples/config/               GITIGNORED. Thesis runs: final_thesis_runs/{base,tiny}_<case>_h<H>.yaml;
                               surrogate: gnn_heterogeneous_gns/{surrogate_opf_case118,twostep_opf_case118_eval}.yaml
```

## Ownership
| File | Owner | Rule |
|---|---|---|
| `tasks/opf_task.py` | co-owned (colleague Matteo-Mazzonelli1) | base `on_test_end` writer UNTOUCHABLE; additive hooks only (`_override_residual_load`, `_after_opf_metrics`) |
| `tasks/{opf_twostep_task,st_forecast_opf_task,constraint_metrics,perbus_residual_dump}.py` | mine | free to rewrite |
| `models/{gnn_heterogeneous_gns,st_gnn_heterogeneous}.py` | mine | free to rewrite |

## Run & validate (capella, sbatch)
```bash
sbatch test_2min.sh                                    # 1-GPU E2E smoke (edit CONFIG inside)
sbatch --time=10:00:00 --export=ALL,SEED=0,CONFIG=examples/config/final_thesis_runs/base_case118_h24.yaml,EXP_NAME=final_thesis_runs,RUN_NAME=base_e2e_case118_h24_s0 train.sh   # E2E, 4-GPU DDP
sbatch train_surrogate.sh                              # surrogate, 3 seeds (job array)
sbatch evaluate.sh                                     # test-only eval from checkpoint
pytest tests/                                          # unit tests
```
- Output: `<mlruns>/<exp_id>/<run_id>/artifacts/test/<network>_metrics.csv` (+ forecast.csv, perbus parquet).
- Report the panel, not just val loss: `Mean Optimality Gap`, `MAE Pd`, `Vm MAE`, `Generator Pg MAE`,
  `Residual P (MAE)`, `qg viol. rate`.
- base needs `--time=10:00:00` (~2.9 min/epoch; default 5:10 reaches only ~107 of 200 epochs).
- Run names are parsed by Thesis_Repo `aggregate_results._ml_identity`: E2E `base_e2e_<case>_h<H>_s<S>`,
  surrogate `surrogate_<case>_<fc>_h<H>_s<S>`. Any other prefix/suffix is skipped (safe for ablations).

## Rules & traps
- Queued SLURM jobs read the working tree **when they start**. Don't edit/switch branches under pending
  jobs; launch from a `git worktree` / code snapshot if it matters.
- `forecast_decoder_mode` is compared `== "per_horizon"`; `per-horizon` silently falls back to shared.
- `data.train_window` is the real training-size knob, not `data.scenarios`. Check params, not run names.
- In MLflow `params/args/...`, an ABSENT key means the code's `getattr` default applied.
- Loss weights give Pd/Vm/Va/Pg exactly 0.25 each by design. Don't rebalance toward Pg "to improve gap".
- Don't report correlations from n<20 runs (signs flip between groups).
- A flat late training curve is usually the LR collapse, not convergence: check `Learning Rate_epoch`.
- Window-normalizer physics fix is guarded by `window_baseMVA is not None`; keep global-norm + surrogate paths byte-identical.
- case500's network key / disk name is `case500_goc`; eval network must match a key in the checkpoint's `normalizer_stats.pt`.
- Temporal datasets skip rebuilding when `processed_raw_files.done` exists: after regenerating raw data,
  delete `processed_temporal/` (check `.done` dates vs `raw/args.log`).

# Findings — gridfm-graphkit-dev (surrogate + E2E arms)

Measured results, retractions, and implementation rationale. Dated; newest facts win.
RETRACTED entries stay so they are not re-proposed. Cross-arm/aggregation findings: `~/Thesis_Repo/docs/findings.md`.

---

## 1. Model design & rationale

### Output bounding & what is predicted vs derived
- Both arms sigmoid-bound only **Vm and Pg**. Va and Qg are free. Vm/Pg violation metrics are therefore
  structurally 0 for both ML arms; informative ones are Qg, thermal, angle.
- Va=0 violations in smoke tests are a genuine physics finding (angle limits not tight at h=1), not a bounding artifact.
- **Surrogate predicts Pg; only Qg is derived (both arms).** `mlp_gen` emits Pg directly, sigmoid-bounded PER
  GENERATOR before `scatter_add` to bus level — that ordering keeps the box limit exact. Bus-level "Pg" in the
  4-col OPF output is only that scatter_add. Qg is the sole physics-solved quantity in either arm.
- Surrogate Qg derivation (`PhysicsDecoderOPF`, `models/utils.py`): `Qg = Q_in + Qd − q_shunt` at PV/REF, 0 at PQ.

### E2E decoder: Qd/Qg derivation
- Bus decoder predicts only Pd, Vm, Va. Qd = Pd × fixed per-bus Q/P ratio, recovered statelessly each forward from
  the lookback window's max-|Pd| snapshot (stable; matches data generation: uniform ratio per bus across scenarios,
  so forecasting Qd independently would be circular).
- Qg derived from reactive balance via the physics layers (`qg_phys = Q_in − q_shunt; qg = qg_phys + qd`, masked to
  PV|REF), consuming the derived Qd DETACHED → Qg MSE trains voltages only, not the load forecast (as in the
  surrogate, where Qd is an input).
- Loss supervises Pd + Vm, Va + derived Qg (`lambda_qg`, default lambda_opf/2 = surrogate parity, since
  **2026-10-08**); excludes Qd. Runs before 2026-10-08 (all `final_thesis_runs` E2E) had NO Qg MSE.
- History (2026-09-21): a July note said E2E predicted Qg as a free head; by 2026-09-21 both arms derived it. Qg RMSE
  numbers from the predict-Qg era (surrogate PQ 0/PV≈47/REF≈110 vs E2E PQ≈0.64/PV≈3/REF≈5) do not describe current runs.

### Dispatch-only E2E (opt-in `model.predict_load: false`)
- No Pd forecast (Pd/Qd output cols NaN); head predicts [Qg, Vm, Va] (Qg 0 at PQ); physics vs TRUE load.
- Only `direct` forecast_mode; no `predict_load_delta`.

### E2E forecast modes (`data.forecast_mode`)
- `direct` (default): whole horizon in one forward. `autoregressive`: 1 step, rolled back into the window.
  `block_autoregressive`: `data.forecast_block` steps per call, rolled back n/block times (block=6, h=24 → 4 calls).
- Model `self.n` = steps-per-call = block for feedback modes (NOT horizon) → decoder + per-horizon heads size to the
  block → every param used each step (no DDP unused-param error).
- One unified `_autoregressive_rollout(batch, block)`; training supervises only the first block. It fixes 3 bugs that
  also broke plain autoregressive once cyclical time features landed: `to_data_list` on non-tensor globals
  (is_normalized/mask_dict); mask_dict lost on round-trip (re-tiled onto sub-batch); fed-back graphs missing the 6
  time-feature cols (re-appended from the datamodule's cyclical table).

### Cyclical time features (default ON)
- XGB's 6 cyclical features (hour/dow/doy sin+cos) broadcast to every bus at every lookback step.
  `data.append_time_features` (default true), `time_feature_start_date`, `time_feature_frequency`.
- Configs keep `input_bus_dim: 15`; `UnmaskedSpatialEncoder` auto-adds `N_TIME_FEATURES` (globals.py, cols 15..20).
- `build_cyclical_time_table()` + `_append_time_features()` in `powergrid_hetero_temporal_dataset.py`; appended AFTER
  normalization. scenario_id == chronological hour offset (verified == load_scenario_idx), anchor 2019-01-01 hourly
  — matches XGB exactly.

### Other opt-ins
- `predict_load_delta` (commit 809a5d1, default off): Pd = last_observed + correction (persistence-relative,
  drift-robust); loss/metrics/denorm stay absolute.
- `data.train_window` (commit c25a08d): caps training to the most-recent K scenarios in `split_dataset_by_time`;
  val/test stay byte-identical (verified).
- `ST_OptimalityGapLoss` (cost-MSE) exists in `training/loss.py` but is NOT in any active config.
- `temporal_decoder: seasonal_skip` (`seasonal_period: 24`): out_k = linear time-mix + zero-init W_s(z[W-24+k-1]) +
  zero-init W_c(terminal).

### Architecture facts
- Encoders are two separate classes (GNS_heterogeneous vs UnmaskedSpatialEncoder) — write "identically configured",
  never "identical"/"shared module". Both 12-layer stacks: 20,003,328 params.
- Totals: E2E 46,243,776 (48 per-horizon heads = 19.1M = 41%); surrogate 20,069,187.
- Versions: torch 2.7.1+cu128, torch_geometric 2.7.0, torch_scatter 2.1.2.
- prune-then-drop: gen=6 / edge=10 dims (not 7/11) because G_ON/B_ON filter inactive elements, then are sliced off.
- TCN auto-depth 4 blocks, receptive field 61 ≥ W=48 (the "12 layers" knob is the spatial GNN).

---

## 2. Metrics semantics

- **Residuals** (`Residual P/Q MAE`): power-balance error after true loads arrive. `_override_residual_load`
  (surrogate) / `target_batch` with true loads (E2E) replace the forecast load before residual computation.
- **Constraint violations**: prediction vs static limits in `batch`; independent of the load override.
- **Residual spaces**: realized (vs true load = operational feasibility) vs forecast (vs own forecast = modeling
  infeasibility). Signed decomposition exact: realized = forecast-space + forecast error; exact-AC realized == pure
  forecast error. MAE is not additive.
- `PerBusResidualAccumulator` writes `<ds>_perbus_residuals_{realized,forecast}.parquet` and
  `<ds>_state_{bus,gen}.parquet` (pred+true pd/vm/va_deg/pg) in the test pass (always devices=1); realized gated
  vs logged "Residual P (MAE)". Backfill job 4288106 (2026-09-24): all gates 0/47 diffs.
- Timing: only the model forward is timed (CUDA-synced, InferenceTimer + `_time_forward`); per-sample divisor
  `_infer_samples` (surrogate num_graphs; E2E B windows × n steps). E2E CSVs before 2026-09-23 divided by windows →
  inflated ×h at h>1. Surrogate GNN training time is not logged.

---

## 3. Physics & reactive power (cross-arm)

- **Physics-loss asymmetry is functional form, NOT weight** (2026-09): nominal weight 0.1 in both arms. Surrogate
  `loss_weights: [0.1, 0.1, 0.8]` maps BY INDEX onto `[LayeredWeightedPhysics, MaskedGenMSE, MaskedBusMSE]`
  (physics=0.1; easy to misread as 0.8). Surrogate physics = `linalg.norm([res_P,res_Q]).mean()` (linear), E2E =
  `mean(res_P**2 + res_Q**2)` (squared) → at 0.01 p.u. residual ~1e-2 vs ~1e-4. Plus per-layer vs output-only.
- **E2E reactive residual is algebraically 0 at PV/REF**: Qg is `Q_in − q_shunt + Qd`, physics scores
  `Qg − Qd + q_shunt − Q_in` → identically 0 at every PV|REF bus (global normalizer → window_scale None). Only PQ
  buses give reactive gradient. Same holds for the surrogate. Scoring vs TRUE Qd instead = pure load error with a
  physics label. Predicting Qg is what restores physics content there.
- **Reactive residual is NOT comparable across derived- vs predicted-Qg models** (2026-10-10, case118 h24, baseline
  s0+s1 vs dispatch-only s0-s2): derived Qg hides Qg error. Baseline Res Q 0.76 MVAr but Qg MAE 9.8, Qg viol max
  ~99; dispatch-only Res Q 1.18 (rise entirely at PV/REF; PQ 0.89 vs 0.90) but Qg MAE 0.80, viol max ~8 — robust
  over seeds. Opt gap/Pg/Vm/Va/Res P unchanged (dropping Pd cost nothing). Thermal viol NOT robust (dispatch
  24k/86k/55k vs base 83k). → Report reactive residual at PQ buses + Qg error/violations.
  - Without Qg MSE (dispatch-only, n=1): Qg MAE 5.1, viol mean 1.2 → the Qg MSE is needed.
- **l2 vs squared physics, same seed s1** (2026-10-10): Res P 2.22→1.92, Res Q 0.72→0.57 (PQ −30%), ~2× baseline seed
  range; Pd MAE/opt gap unchanged; Vm/Va MAE +11-16% worse. n=1, promising. The l2 term is ~10× larger at the same
  weight → form and effective weight changed together (a matched-weight squared arm would separate them).
  RETRACTED (2026-10-10): "l2 improves Pd" and "dispatch-only thermal −70%" were seed artifacts.
- Seed noise (base, s0 vs s1): gap ±1%, Res P 7%, Res Q 10%, Pd MAE 23%.
- **Vm bounding destroys the violation signal at PQ buses**: correct at PV/REF (setpoints), wrong at PQ (voltage is a
  consequence). Fix (NOT implemented): bound only where `mask_dict["PV"] | mask_dict["REF"]`.
- Tradeoff framing: predict Qg = best Qg accuracy + box enforceable, worst balance residual; derive = residual 0 at gen
  buses but Qg unbounded and no gen-bus reactive gradient — error is routed between equality and inequality.

---

## 4. Training & evaluation (E2E ST-GNN, 2026-09-22)

- **Validation loss is a legitimate selection criterion.** RETRACTED: "val loss anti-correlates with optimality gap"
  (one group, n=8-11). Spearman(best val, gap) across homogeneous groups: +0.44 (n=23, p~0.035), −0.56 (n=11),
  +0.32 (n=9), +0.68 (n=7), +0.14 (n=6). Within small groups gap and Pg MAE sometimes disagreed in sign.
- Loss weights `[0.75, 0.25, 0.1]` over `[BusMSE, GenMSE, Physics]` with `lambda_load 0.333` / `lambda_opf 0.667` →
  Pd/Vm/Va/Pg 0.25 each, by design (the per-variable E2E-vs-2-step comparison rests on this).
- **LR collapse**: `ReduceLROnPlateau(factor=0.7, patience=5)` on "Validation loss" (`tasks/base_task.py`), threshold
  1e-4, cooldown 0, no min_lr. Observed 1e-4 → 1.1e-7…6.8e-7 (~19 decays / 200 epochs). Never tuned: `lr_HPO_h48`
  swept only initial lr; lr_patience 5 in 179/181 configs, lr_decay 0.7 in all. Don't recommend more capacity for
  "underfitting" — the optimizer stopped. Relative A/B comparisons still hold (shared schedule).
- `AdamW` has no `weight_decay` → PyTorch default 0.01 on all params incl. LayerNorm/bias; decays with lr.
- Walltime: base ~2.9 min/epoch → 5:10 reaches ~107 epochs; 200 epochs needs ~10 h. tiny 200 in ~3 h.
  `callbacks.patience: 100` → early stopping effectively never fires.
- **Fit regime depends on size**: base (hs=48) overfits, final val/train 3.0-4.3 on h24; tiny (hs=12) underfits,
  0.8-1.9. Gap grows with horizon (h1 ~1.01, h6 ~1.63, h24 ~3.41). hs=24/32 never run.
- Overfitting localizes to the cyclical calendar channels: `*_no_temporal` config is ONLY
  `append_time_features: false` (does NOT remove the temporal encoder) and drops the ratio 3.41 → 1.23, but its gap
  is worse (4.955 vs 4.295) → not a reason to disable them. Mechanism (plausible): temporal_split puts val/test later,
  periodic channels let a big model memorize the training period.
- With `temporal_decoder: terminal` + `shared` heads + no step embedding all n steps get identical input+weights →
  same prediction n times. Guarded in code.

### Temporal bottleneck diagnosis (2026-07-18, 79 case118 runs)
- corr(gap, rmse_pg)=0.90, corr(gap, rmse_pd)=0.60; true-load surrogate h24 gap 0.84% vs best E2E 4.22%; E2E h1 gap
  ~0.55% → forecasting is the bottleneck, not the dispatch head.
- Per-step MASE of best TCN: t+1..t+12 ≈ 0.28-0.36, t+24 = 0.73.
- Failed transformer runs were confounded (cross_attention decoder + no harmonic weighting; batch 4 vs 16). Clean-ish
  variants 4.7-5.2% vs TCN 4.22%.

---

## 5. Settled / low-leverage — don't re-litigate

- **W→n temporal decoder axis**: matched comparisons show terminal-state broadcast ≈ `Linear(W->n)` (0.9-2% apart,
  within spread; e.g. 0.01473 vs 0.01460 at hs12, 0.01446 vs 0.01416 at hs48). TCN RF (61) > W (48).
  `Linear(W->n)` is 2,352 params of ~28-46M. Caveats: only 15/356 runs log `model.temporal_decoder` (rest
  date-inferred); `seasonal_skip` = 1 truncated run (177/200, no matched control) → not a clean negative;
  explicit `temporal_decoder="terminal"` has 0 runs.
- **More training data does not help** (matched `train_window`): tiny 26280→3.799 vs None→3.839; base 26280→3.845 vs
  None→3.934.
- **Window normalizer is not better than global MVA**: the ~0.9pp advantage was a dataset artifact; like-for-like run
  gives 4.476, inside the MVA spread (4.095-5.105).
- Cross-folder gap comparisons (`final_thesis_runs` 4.10-5.56 vs `24yr` 3.79-4.16) are on different test periods.

---

## 6. Data / MLflow archive integrity

- **Contaminated runs**: `getattr(args.model, "temporal_decoder", ...)` defaulted to `cross_attention` for ~33.5 h
  (2026-05-12 08:54 → 05-13 18:23, commits 5296a68 → 876fe24). The three `*_noAttn_noRes` runs in experiment
  `step_embedding` (0374a2f7fc5a, e6fab2327d4a, b09d87774525) almost certainly ran cross-attention → discard any
  attention-vs-no-attention conclusion from it. No other axis ever flipped.
- Runs launch from dirty trees (one started 2 days before its own commit) → date-based architecture inference is
  unreliable. Several runs named `3yr_*` actually ran `train_window=None`.
- `horizon_weighting`: 238 runs uniform (key absent), 118 harmonic; first harmonic 2026-05-18.
- **E2E case500 stale processed_temporal** (2026-10-10): `case500_goc/processed_temporal/` built 2026-06-21 from an
  older raw generation: 22072 vs 26280 scenarios, 624 gaps in load_scenario_idx (max 32 h), loads ~1.22-1.25× lower
  than current raw. Cause: `HeteroGridDatasetDisk.process()` skips rebuild when `processed_raw_files.done` exists.
  Every E2E case500 metric (all horizons/seeds) trained+tested on old data (3300 vs ~3931 test windows, misaligned
  calendar features). case14/case118 OK. New runs: `base_e2e_case500_goc_h{H}_s0`; archive the old
  `base_e2e_case500_h{H}_s0` before aggregating.
- **Surrogate normalizer key mismatch** (2026-08-18): eval restores `normalizer_stats.pt` only if
  `network in saved_stats` and `fit_on_train`; else it refits on the (empty, `eval_full_as_test`) train split →
  `np.percentile([])` IndexError. Root cause was `train_surrogate.sh` training every case on case118 (CASE only
  changed the name); fixed by sed-injecting `networks: [$CASE]`. case500 key is `case500_goc`.
- Block-autoregressive ablation runs share run names with E2E case500 h6/h24 (exp 909650501617624148) → resolvers
  pinned to E2E exp 375695290152400764.

---

## 7. 24yr concept-drift study (branch exp/e2e-24yr-drift)

- "24yr trained worse than 3yr" was confounded three ways:
  - Data realization: 2019-2021 and 2002-2025 sets differ — 0/300 sampled 3yr load vectors match any 24yr scenario
    (nearest L-inf ≥ 10.75 MW); 3yr ~30% hotter (mean 2263 vs 1732 MW; 3yr max 4097 > 24yr max 3665).
  - Normalizer: 3yr global MVA vs 24yr window MVA.
  - Test period: 24yr tests on 2022-2025, 3yr on ~2021.
- Drift on 24yr: load level +24% (step-ups ~2018, ~2022), shape stable (COV ~0.23 flat; first-pass proxy only).
- **Window-normalizer physics bug — FIXED (commit 878e46a)**: power is window-p.u. (b_w), admittances/shunts are
  global-p.u. (b_s) → mixed bases give spurious residual (old runs ×8 vs global-norm, gap unchanged). Old post-hoc
  rescale of the final residual was wrong. Fix: `ST_PhysicsForecastLoss` lifts Pd,Qd,Qg,Pg to global p.u. (× b_w/b_s)
  before physics; model Qg derivation rebases (Q_in − q_shunt) to window p.u. (× b_s/b_w); guarded by
  `window_baseMVA is not None`; threaded `task._window_scale → model.forward(window_scale=...)`. Validated offline:
  window-norm ≈ global-norm on 3yr.
- Clean drift test: `examples/config/24yr_case118_gns/drift_A_fullhist_windownorm.yaml` vs
  `drift_B_recent3yr_windownorm.yaml` (train_window=26280), byte-identical test set.
- Speed: 1-GPU 24yr times out (~35 epochs / 8 h); preload ~13 min. Fast configs `24yr_118_fast_h24{,_delta}.yaml`
  (4 layers, hidden 16, mlp 256) — cut depth, not just width.

"""
OptimalPowerFlow two-step task (OPF surrogate on forecasted loads).

- Almost identical to OptimalPowerFlowTask for the forward pass and every metric.
- ONE difference: the power-balance residual is scored against the TRUE realized
  load, not the model's (forecasted) input load.

Why: in the 2-step pipeline the surrogate is fed a forecasted load and predicts a
dispatch; its operational feasibility should be measured against the load that
actually occurs.
-> matches Thesis_Repo compare.py (uses Pd_true) and the E2E model (evaluates vs
   the true-future target load). The model still sees the forecast in the forward
   pass; only the residual uses the true load.

Contract: data["bus"].true_load = [N_bus, 2] (Pd, Qd) physical units; per-bus node attr.

Metrics: emits the canonical metrics.csv (compare.py-aligned names + the
constraint-violation quintet) so this arm is directly comparable to the exact
(compare.py) and E2E arms. The base RMSE.csv (per-bus-type) is left untouched.
"""

import os

import torch
import torch.distributed as dist
import pandas as pd
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger

from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.datasets.globals import PD_H, QD_H
from gridfm_graphkit.models.utils import ComputeBranchFlow
from gridfm_graphkit.tasks.perbus_residual_dump import PerBusResidualAccumulator
from gridfm_graphkit.tasks.constraint_metrics import (
    ConstraintViolationAccumulator,
    per_type_from_opf_batch,
    canonical_scalar_rows,
)


@TASK_REGISTRY.register("OptimalPowerFlowTwoStep")
class OptimalPowerFlowTwoStepTask(OptimalPowerFlowTask):
    """OPF surrogate on forecasted loads; residual scored vs true realized load."""

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)
        self._branch_flow = ComputeBranchFlow()
        self._viol_acc = {}
        # System-wide load/dispatch totals (physical MW), accumulated per batch.
        self._totals = {}
        # Snapshot of the current batch's forecast/true Pd sums, set in
        # _override_residual_load (which has both loads in hand) and consumed in
        # _after_opf_metrics (which knows dataloader_idx).
        self._pending_pd_pred = 0.0
        self._pending_pd_true = 0.0
        self._pending_forecast_bus = None  # per-bus forecast load snapshot (set per batch)
        self._perbus_acc = None

    def on_test_start(self):
        # Fresh per-network violation accumulators + totals for this test run.
        n = len(self.args.data.networks)
        self._viol_acc = {i: ConstraintViolationAccumulator() for i in range(n)}
        self._totals = {
            i: {"pd_pred": 0.0, "pd_true": 0.0, "pg_pred": 0.0, "pg_true": 0.0}
            for i in range(n)
        }
        self._perbus_acc = PerBusResidualAccumulator(n)

    def _override_residual_load(self, batch):
        # replace (forecast) load in x with the true realized load, in place,
        # so _compute_opf_metrics scores the power-balance residual against reality
        true_load = batch["bus"].true_load  # [N_bus, 2] = (Pd_true, Qd_true), physical
        # Snapshot Pd sums BEFORE the overwrite: x still holds the forecast input.
        # (batch is already inverse_transform'd, so both are physical MW.)
        self._pending_pd_pred = batch.x_dict["bus"][:, PD_H].sum().item()
        self._pending_pd_true = true_load[:, 0].sum().item()
        # Per-bus forecast load snapshot (before overwrite) for the forecast-space residual.
        self._pending_forecast_bus = torch.stack(
            [batch.x_dict["bus"][:, PD_H], batch.x_dict["bus"][:, QD_H]], dim=1
        ).detach().clone()
        batch.x_dict["bus"][:, PD_H] = true_load[:, 0]
        batch.x_dict["bus"][:, QD_H] = true_load[:, 1]

    def _after_opf_metrics(self, output, target, batch, dataloader_idx=0):
        # Accumulate canonical constraint-violation stats for this batch.
        # Violations use the predicted dispatch vs static limits — independent of
        # the load override above.
        per_type, num_scen = per_type_from_opf_batch(output, batch, self._branch_flow)
        self._viol_acc[dataloader_idx].update(per_type, num_scen)

        # System-wide totals (physical MW). pd_pred = forecast input load;
        # pd_true = realized load; pg_true = OPF-on-true-loads target dispatch;
        # pg_pred = surrogate dispatch on forecasted loads. (output/target already
        # inverse_output/inverse_transform'd by the base test_step.)
        t = self._totals[dataloader_idx]
        t["pd_pred"] += self._pending_pd_pred
        t["pd_true"] += self._pending_pd_true
        t["pg_pred"] += output["gen"].sum().item()
        t["pg_true"] += target["gen"].sum().item()

        self._perbus_acc.accumulate(output, batch, dataloader_idx,
                                    load_pred=self._pending_forecast_bus)

    @rank_zero_only
    def _canonical_metrics_dir(self):
        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir, self.logger.experiment_id,
                self.logger.run_id, "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir
        return os.path.join(artifact_dir, "test")

    def _reduce_totals(self):
        # DDP-sum the per-network totals so they are grid-total consistent with
        # the violation quintet (no-op under single-device eval / no DDP).
        if not (dist.is_available() and dist.is_initialized()):
            return
        for t in self._totals.values():
            vec = torch.tensor(
                [t["pd_pred"], t["pd_true"], t["pg_pred"], t["pg_true"]],
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            t["pd_pred"], t["pd_true"], t["pg_pred"], t["pg_true"] = vec.tolist()

    def on_test_end(self):
        # Reduce accumulators across DDP ranks BEFORE the rank-zero-only writer.
        for acc in self._viol_acc.values():
            acc.reduce_across_ranks(self.device)
        self._reduce_totals()
        # Base writes RMSE.csv (kept) + its own metrics.csv (overwritten below).
        super().on_test_end()
        self._write_canonical_metrics()

    @rank_zero_only
    def _write_canonical_metrics(self):
        # Group callback_metrics by dataset (same convention as the base writer).
        grouped = {}
        for full_key, value in self.trainer.callback_metrics.items():
            if "/" not in full_key:
                continue
            ds, metric = full_key.split("/", 1)
            grouped.setdefault(ds, {})[metric] = (
                value.item() if hasattr(value, "item") else value
            )

        test_dir = self._canonical_metrics_dir()
        os.makedirs(test_dir, exist_ok=True)

        # Forward-only inference timing (populated by shared_step via the
        # InferenceTimer callback; per-rank, i.e. rank 0 here). No training time:
        # surrogate eval is a standalone `evaluate` run, so .fit() never runs.
        infer_total = float(getattr(self, "_infer_time_s", 0.0))
        infer_samples = int(getattr(self, "_infer_samples", 0))
        i_time_val = infer_total if infer_samples > 0 else " "
        i_time_per_sample = infer_total / infer_samples if infer_samples > 0 else " "

        for idx, dataset in enumerate(self.args.data.networks):
            rows = canonical_scalar_rows(grouped.get(dataset, {}))
            rows += self._viol_acc[idx].finalize_rows(prefix="")
            t = self._totals[idx]
            rows += [
                {"Metric": "total_pd_true", "Value": t["pd_true"], "Unit": "MW"},
                {"Metric": "total_pd_pred", "Value": t["pd_pred"], "Unit": "MW"},
                {"Metric": "total_pg_true", "Value": t["pg_true"], "Unit": "MW"},
                {"Metric": "total_pg_pred", "Value": t["pg_pred"], "Unit": "MW"},
                {"Metric": "Inference time model-only (s)", "Value": i_time_val, "Unit": "s"},
                {"Metric": "Inference time model-only per sample (s)", "Value": i_time_per_sample, "Unit": "s"},
            ]
            path = os.path.join(test_dir, f"{dataset}_metrics.csv")
            pd.DataFrame(rows).to_csv(path, index=False)

            self._perbus_acc.write(idx, dataset, test_dir, grouped.get(dataset, {}))

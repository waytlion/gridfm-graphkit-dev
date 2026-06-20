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
"""

from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.datasets.globals import PD_H, QD_H


@TASK_REGISTRY.register("OptimalPowerFlowTwoStep")
class OptimalPowerFlowTwoStepTask(OptimalPowerFlowTask):
    """OPF surrogate on forecasted loads; residual scored vs true realized load."""

    def _override_residual_load(self, batch):
        # replace (forecast) load in x with the true realized load, in place,
        # so _compute_opf_metrics scores the power-balance residual against reality
        true_load = batch["bus"].true_load  # [N_bus, 2] = (Pd_true, Qd_true), physical
        batch.x_dict["bus"][:, PD_H] = true_load[:, 0]
        batch.x_dict["bus"][:, QD_H] = true_load[:, 1]

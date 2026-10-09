"""

What?
    - recomputes per-bus P/Q residuals in test pass  
Why?: 
    - writes parquets for the thesis error-distribution plot

How?
    - used by surrogate (OptimalPowerFlowTwoStepTask) + E2E (ST_ForecastOPFTask), single-device test pass only
    - realized-space dump is gated against the base task's already-logged "Residual P (MAE)" (safety check)
"""

import os

import numpy as np
import torch
import pandas as pd

from gridfm_graphkit.datasets.globals import PD_H, QD_H
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)


class PerBusResidualAccumulator:
    """accumulates per-(bus x scenario) signed P/Q residuals across test batches, then writes parquet(s) per dataset.
    """

    def __init__(self, n_networks: int):
        self._recs = {i: [] for i in range(n_networks)}        # dataloader_idx -> per-batch records
        self._scen_offset = {i: 0 for i in range(n_networks)}  # running scenario counter per dataset
        self._branch_flow = ComputeBranchFlow()
        self._node_inj = ComputeNodeInjection()
        self._node_res = ComputeNodeResiduals()

    def accumulate(self, output, batch, dataloader_idx, load_pred, target):
        """Recompute + stash SIGNED per-bus P/Q residuals (MW/MVAr) computed using true load (== in the REALIZED space)

        Args:
            - load_pred ([num_bus, 2] = predicted Pd, Qd, physical MW)

        Notes:
            - ALSO stashes the FORECAST-space residual (dispatch vs its own forecast load)
                -> FORECAST-space residual == the intrinsic modeling infeasibility.
            - Only the load term differs between spaces FORECAST and REALIZED
                -> (Pg, Vm, Va), are always predicted
                -> so branch flow, node injection is the same in both spaces
                -> so FORECAST-space residual can be computed as an exact delta: res_forecast = res_realized - (load_forecast - load_true)
        """
        num_bus = batch.x_dict["bus"].size(0)
        bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]

        # Recompute the residual: branch flows -> nodal injection -> power-balance mismatch.
        with torch.no_grad():
            Pft, Qft = self._branch_flow(output["bus"], bus_edge_index, bus_edge_attr)
            P_in, Q_in = self._node_inj(Pft, Qft, bus_edge_index, num_bus)
            res_P, res_Q = self._node_res(P_in, Q_in, output["bus"], batch.x_dict["bus"])

        # Per-graph bus count from the batch itself (not hardcoded -> any case size works).
        bus_graph = batch.batch_dict["bus"]
        G = int(bus_graph.max().item()) + 1
        N_bus = int((bus_graph == 0).sum().item())

        # exact delta (see docstring): res_forecast = res_realized - (load_forecast - load_true)
        pd_true = batch.x_dict["bus"][:, PD_H]
        qd_true = batch.x_dict["bus"][:, QD_H]
        res_P_fc = (res_P - (load_pred[:, 0] - pd_true)).detach().cpu().view(G, N_bus)
        res_Q_fc = (res_Q - (load_pred[:, 1] - qd_true)).detach().cpu().view(G, N_bus)
        pd_fc = load_pred[:, 0].detach().cpu().view(G, N_bus)

        res_P = res_P.detach().cpu().view(G, N_bus)
        res_Q = res_Q.detach().cpu().view(G, N_bus)
        pd_true = pd_true.detach().cpu().view(G, N_bus)

        out_b = output["bus"][:, :3].detach().cpu().view(G, N_bus, 3).clone()
        tgt_b = target["bus"][:, :3].detach().cpu().view(G, N_bus, 3).clone()
        out_b[..., 1] = torch.rad2deg(out_b[..., 1])
        tgt_b[..., 1] = torch.rad2deg(tgt_b[..., 1])
        out_g = output["gen"].detach().cpu().view(G, -1)
        tgt_g = target["gen"].detach().cpu().view(G, -1)

        # One record per scenario (graph) in this batch.
        base = self._scen_offset[dataloader_idx]
        for g in range(G):
            self._recs[dataloader_idx].append({
                "scenario": base + g,
                "res_p_mw": res_P[g].numpy(),
                "res_q_mvar": res_Q[g].numpy(),
                "pd_true_mw": pd_true[g].numpy(),
                "res_p_mw_fc": res_P_fc[g].numpy(),
                "res_q_mvar_fc": res_Q_fc[g].numpy(),
                "pd_forecast_mw": pd_fc[g].numpy(),
            })
            # per-bus/per-gen state (pred vs OPF-on-true-load target) for the error-distribution plots
            self._recs[dataloader_idx][-1].update({
                "pd_pred": pd_fc[g].numpy(), "pd_true": pd_true[g].numpy(),
                **{f"{k}_{s}": v[g].numpy() for s, src in (("pred", out_b), ("true", tgt_b)) for k, v in
                   (("vm", src[:, :, 0]), ("va_deg", src[:, :, 1]), ("pg", src[:, :, 2]))},
                "gen_pg_pred": out_g[g].numpy(), "gen_pg_true": tgt_g[g].numpy(),
            })
        self._scen_offset[dataloader_idx] = base + G

    def write(self, idx, dataset, test_dir, ds_metrics):
        """Write both spaces to parquet; gate the realized dump against the already-logged metric.

        <dataset>_perbus_residuals_realized.parquet : vs true load (operational feasibility)
        <dataset>_perbus_residuals_forecast.parquet  : vs own forecast load (modeling infeasibility)
        Schema: [scenario, bus, res_p_mw, res_q_mvar, pd_true_mw|pd_forecast_mw]
        """
        recs = self._recs.get(idx, [])
        if not recs:
            return
        n_bus = len(recs[0]["res_p_mw"])

        def _to_df(res_key, resq_key, load_key):
            rows = [(r["scenario"], b, float(r[res_key][b]), float(r[resq_key][b]), float(r[load_key][b]))
                    for r in recs for b in range(n_bus)]
            return pd.DataFrame(rows, columns=["scenario", "bus", "res_p_mw", "res_q_mvar", load_key])

        # Realized space: regression gate against the already-logged "Residual P (MAE)"
        # (= OptimalPowerFlowTask's "Active Power Loss"). Both are mean(|res_P|) over the
        # same physics, computed independently in two places -- a mismatch means a
        # unit/reshape/convention bug crept into THIS accumulator; fail loudly, now.
        df = _to_df("res_p_mw", "res_q_mvar", "pd_true_mw")
        logged_mae = float(ds_metrics["Active Power Loss"])
        dumped_mae = df["res_p_mw"].abs().mean()
        if abs(dumped_mae - logged_mae) > 1e-3 * max(1.0, abs(logged_mae)):
            raise RuntimeError(
                f"[{dataset}] per-bus residual MAE {dumped_mae:.6f} != already-logged "
                f"Residual P (MAE) {logged_mae:.6f} MW -> unit/convention drift.")
        print(f"[{dataset}] per-bus residual dump: {len(recs)} scenarios x {n_bus} buses | "
              f"realized |res_p| MAE={dumped_mae:.4f} MW (logged {logged_mae:.4f})")
        df.to_parquet(os.path.join(test_dir, f"{dataset}_perbus_residuals_realized.parquet"), index=False)

        # Forecast space: the method's intrinsic modeling infeasibility (no true load involved).
        dffc = _to_df("res_p_mw_fc", "res_q_mvar_fc", "pd_forecast_mw")
        print(f"    + forecast-space (modeling infeasibility): |res_p| MAE={dffc['res_p_mw'].abs().mean():.4f} MW")
        dffc.to_parquet(os.path.join(test_dir, f"{dataset}_perbus_residuals_forecast.parquet"), index=False)

        # State dumps: [scenario, bus, <var>_pred, <var>_true] and [scenario, gen, pg_pred, pg_true]
        bus_cols = ["pd", "vm", "va_deg", "pg"]
        n_gen = len(recs[0]["gen_pg_pred"])
        scen = np.repeat([r["scenario"] for r in recs], n_bus)
        bus = pd.DataFrame({"scenario": scen, "bus": np.tile(np.arange(n_bus), len(recs)),
                            **{f"{c}_{s}": np.concatenate([r[f"{c}_{s}"] for r in recs])
                               for c in bus_cols for s in ("pred", "true")}})
        gen = pd.DataFrame({"scenario": np.repeat([r["scenario"] for r in recs], n_gen),
                            "gen": np.tile(np.arange(n_gen), len(recs)),
                            **{f"pg_{s}": np.concatenate([r[f"gen_pg_{s}"] for r in recs]) for s in ("pred", "true")}})
        rmse = lambda d, c: float(np.sqrt(((d[f"{c}_pred"] - d[f"{c}_true"]) ** 2).mean()))
        print(f"    + state dump: gen Pg RMSE={rmse(gen, 'pg'):.4f} MW | bus Vm RMSE={rmse(bus, 'vm'):.5f} "
              f"Va RMSE={rmse(bus, 'va_deg'):.4f} deg Pd RMSE={rmse(bus, 'pd'):.4f} MW")
        bus.to_parquet(os.path.join(test_dir, f"{dataset}_state_bus.parquet"), index=False)
        gen.to_parquet(os.path.join(test_dir, f"{dataset}_state_gen.parquet"), index=False)

"""
ForecastOPF Task - One-step-ahead OPF forecasting (t → t+1)

Key differences from OptimalPowerFlowTask:
1. Predicts ALL dynamic features: [Pd, Qd, Qg, Vm, Va] (5D bus) + [Pg] (1D gen)
2. shared_step() computes MSE on all 5 features (parent's MaskedBusMSE only uses 2)
3. test_step() adds load forecast MAE metrics (Pd, Qd)
4. Converts output to OPF format [Vm, Va, Pg, Qg] to reuse physics/cost metrics

Inheritance: OptimalPowerFlowTask → ReconstructionTask → BaseTask
"""

from gridfm_graphkit.datasets.globals import (
    #NEW: Load prediction indices (forecast-specific)
    PD_H,     
    QD_H, 
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    MIN_QG_H,
    MAX_QG_H, 
    # Output feature indices
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
    # Generator feature indices
    PG_H,
    C0_H,
    C1_H,
    C2_H,
    # Edge feature indices
    ANG_MIN,
    ANG_MAX,
    RATE_A,
)
from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.utils import (
    plot_correlation_by_node_type,
    plot_residuals_histograms,
    residual_stats_by_type,
)
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
)
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch.loggers import MLFlowLogger
import numpy as np
import os
import pandas as pd


@TASK_REGISTRY.register("ForecastOPF")
class ForecastOPFTask(OptimalPowerFlowTask):
    """
    Forecast OPF Task: Predict complete OPF state at t+1 given state at t.
    
    Output format: bus features are [Pd, Qd, Qg, Vm, Va] (5 features)
                   gen features are [Pg] (1 feature)
    
    Inherits from OptimalPowerFlowTask to reuse:
    - Physics calculations (branch flows, node injections, residuals)
    - Constraint violation checks (Qg limits, thermal, angle)
    - Cost computation and optimality gap
    - Plotting infrastructure
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

    def _convert_to_opf_format(self, output, batch):
        """
        Convert ForecastOPF output to OPF format for metrics reuse.
        
        ForecastOPF output: [Pd, Qd, Qg, Vm, Va] (5 features)
        OPF expected:       [Vm, Va, Pg_agg, Qg] (4 features)
        
        Args:
            output: ForecastOPF predictions {"bus": [N, 5], "gen": [M, 1]}
            batch: Batch with edge indices for aggregation
            
        Returns:
            output_opf: OPF-format predictions {"bus": [N, 4], "gen": [M, 1]}
        """
        pred_bus = output["bus"]  # [num_bus, 5] 
        pred_gen = output["gen"]  # [num_gen, 1] 
        
        #SAME: Aggregate generator Pg to buses
        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]
        num_bus = pred_bus.size(0)
        agg_pg = scatter_add(
            pred_gen,
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )
        
        # Reorder to OPF format: [Vm, Va, Pg_agg, Qg]
        output_opf = torch.stack(
            [
                pred_bus[:, 3],        # Vm (index 3 in ForecastOPF)
                pred_bus[:, 4],        # Va (index 4)
                agg_pg.squeeze(),      # Pg aggregated from generators
                pred_bus[:, 2],        # Qg (index 2)
            ],
            dim=1,
        )
        
        return {"bus": output_opf, "gen": pred_gen}

    def _convert_target_to_opf_format(self, batch):
        """
        Convert ForecastOPF targets to OPF format.
        
        Same logic as _convert_to_opf_format but for ground truth.
        """
        target_bus = batch.y_dict["bus"]  # [num_bus, 5]
        target_gen = batch.y_dict["gen"]  # [num_gen, 1]
        
        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]
        num_bus = target_bus.size(0)
        agg_pg = scatter_add(
            target_gen,
            gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )
        
        target_opf = torch.stack(
            [
                target_bus[:, 3],      # Vm
                target_bus[:, 4],      # Va
                agg_pg.squeeze(),      # Pg
                target_bus[:, 2],      # Qg
            ],
            dim=1,
        )
        
        return {"bus": target_opf, "gen": target_gen}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Add load forecast metrics + convert format for OPF metrics.
        
        Structure:
        1. Get predictions and denormalize (SAME as OPF)
        2. NEW: Log load forecast accuracy (MAE for Pd, Qd)
        3. NEW: Convert output/target to OPF format
        4. REUSE: All OPF metrics (copy from opf_task.py)
        5. SAME: Store outputs for plotting
        """
        #1: Prediction and Denormalization (SAME as OPF)
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        #2: NEW - Load Forecast Metrics
        pred_bus = output["bus"]  # [Pd, Qd, Qg, Vm, Va]
        target_bus = batch.y_dict["bus"]
        
        # Compute MAE for load predictions
        mae_Pd = torch.mean(torch.abs(pred_bus[:, 0] - target_bus[:, 0]))
        mae_Qd = torch.mean(torch.abs(pred_bus[:, 1] - target_bus[:, 1]))
        
        # Also compute MAE for voltage and Qg (forecast-specific)
        mae_Vm = torch.mean(torch.abs(pred_bus[:, 3] - target_bus[:, 3]))
        mae_Va = torch.mean(torch.abs(pred_bus[:, 4] - target_bus[:, 4]))
        mae_Qg = torch.mean(torch.abs(pred_bus[:, 2] - target_bus[:, 2]))
        mae_Pg = torch.mean(torch.abs(output["gen"] - batch.y_dict["gen"]))
        
        # Log forecast-specific metrics
        self.log(
            f"{dataset_name}/MAE Pd (MW)",
            mae_Pd,
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.log(
            f"{dataset_name}/MAE Qd (MVar)",
            mae_Qd,
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.log(
            f"{dataset_name}/MAE Vm (p.u.)",
            mae_Vm,
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.log(
            f"{dataset_name}/MAE Va (rad)",
            mae_Va,
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.log(
            f"{dataset_name}/MAE Qg (MVar)",
            mae_Qg,
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.log(
            f"{dataset_name}/MAE Pg (MW)",
            mae_Pg,
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # 3: NEW - Convert to OPF Format 
        output_opf = self._convert_to_opf_format(output, batch)
        target_opf = self._convert_target_to_opf_format(batch)

        #4: REUSE - All OPF Metrics (copy from opf_task.pyy)
        # Physics calculations
        branch_flow_layer = ComputeBranchFlow()
        node_injection_layer = ComputeNodeInjection()
        node_residuals_layer = ComputeNodeResiduals()

        num_bus = batch.x_dict["bus"].size(0)
        bus_edge_index = batch.edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = batch.edge_attr_dict[("bus", "connects", "bus")]
        _, gen_to_bus_index = batch.edge_index_dict[("gen", "connected_to", "bus")]

        # Generator cost computation (SAME as OPF lines 69-85)
        mse_PG = F.mse_loss(
            output["gen"],
            batch.y_dict["gen"],
            reduction="none",
        ).mean(dim=0)
        c0 = batch.x_dict["gen"][:, C0_H]
        c1 = batch.x_dict["gen"][:, C1_H]
        c2 = batch.x_dict["gen"][:, C2_H]
        target_pg = batch.y_dict["gen"].squeeze()
        pred_pg = output["gen"].squeeze()
        gen_cost_gt = c0 + c1 * target_pg + c2 * target_pg**2
        gen_cost_pred = c0 + c1 * pred_pg + c2 * pred_pg**2

        gen_batch = batch.batch_dict["gen"]

        cost_gt = scatter_add(gen_cost_gt, gen_batch, dim=0)
        cost_pred = scatter_add(gen_cost_pred, gen_batch, dim=0)

        optimality_gap = torch.mean(torch.abs((cost_pred - cost_gt) / cost_gt * 100))

        # CHANGED: Use converted OPF format for physics calculations
        # Branch flow computation (SAME logic as OPF)
        Pft, Qft = branch_flow_layer(
            output_opf["bus"],  # Use OPF format [Vm, Va, Pg, Qg]
            bus_edge_index,
            bus_edge_attr,
        )
        
        # Branch thermal violations (SAME as OPF lines)
        Sft = torch.sqrt(Pft**2 + Qft**2)
        branch_thermal_limits = bus_edge_attr[:, RATE_A]
        branch_thermal_excess = F.relu(Sft - branch_thermal_limits)

        num_edges = bus_edge_index.size(1)
        half_edges = num_edges // 2
        forward_excess = branch_thermal_excess[:half_edges]
        reverse_excess = branch_thermal_excess[half_edges:]

        mean_thermal_violation_forward = torch.mean(forward_excess)
        mean_thermal_violation_reverse = torch.mean(reverse_excess)

        # Branch angle violations (SAME as OPF)
        angle_min = bus_edge_attr[:, ANG_MIN]
        angle_max = bus_edge_attr[:, ANG_MAX]

        bus_angles = output_opf["bus"][:, VA_OUT]
        from_bus = bus_edge_index[0]
        to_bus = bus_edge_index[1]
        angle_diff = torch.abs(bus_angles[from_bus] - bus_angles[to_bus])

        angle_excess_low = F.relu(angle_min - angle_diff)
        angle_excess_high = F.relu(angle_diff - angle_max)
        branch_angle_violation_mean = (
            torch.mean(angle_excess_low + angle_excess_high) * 180.0 / torch.pi
        )

        # Node injections and residuals (SAME as OPF)
        P_in, Q_in = node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        residual_P, residual_Q = node_residuals_layer(
            P_in,
            Q_in,
            output_opf["bus"],  # Use OPF format
            batch.x_dict["bus"],
        )

        # Qg violation checks (SAME as OPF)
        Qg_pred = output_opf["bus"][:, QG_OUT]
        Qg_max = batch.x_dict["bus"][:, MAX_QG_H]
        Qg_min = batch.x_dict["bus"][:, MIN_QG_H]

        mask_Qg_violation = (Qg_pred > Qg_max) | (Qg_pred < Qg_min)

        bus_batch = batch.batch_dict["bus"]

        mask_PQ = batch.mask_dict["PQ"]
        mask_PV = batch.mask_dict["PV"]
        mask_REF = batch.mask_dict["REF"]

        Qg_over = F.relu(Qg_pred - Qg_max)
        Qg_under = F.relu(Qg_min - Qg_pred)
        Qg_violation_amount = Qg_over + Qg_under

        mean_Qg_violation_PV = Qg_violation_amount[mask_PV].mean()
        mean_Qg_violation_REF = Qg_violation_amount[mask_REF].mean()

        # Store outputs for plotting (SAME as OpF)
        if self.args.verbose:
            mean_res_P_PQ, max_res_P_PQ = residual_stats_by_type(
                residual_P,
                mask_PQ,
                bus_batch,
            )
            mean_res_Q_PQ, max_res_Q_PQ = residual_stats_by_type(
                residual_Q,
                mask_PQ,
                bus_batch,
            )

            mean_res_P_PV, max_res_P_PV = residual_stats_by_type(
                residual_P,
                mask_PV,
                bus_batch,
            )
            mean_res_Q_PV, max_res_Q_PV = residual_stats_by_type(
                residual_Q,
                mask_PV,
                bus_batch,
            )

            mean_res_P_REF, max_res_P_REF = residual_stats_by_type(
                residual_P,
                mask_REF,
                bus_batch,
            )
            mean_res_Q_REF, max_res_Q_REF = residual_stats_by_type(
                residual_Q,
                mask_REF,
                bus_batch,
            )
            
            # CHANGED: Store both original and OPF format for plotting
            self.test_outputs[dataloader_idx].append(
                {
                    "dataset": dataset_name,
                    "pred": output_opf["bus"].detach().cpu(),  # OPF format for plots
                    "target": target_opf["bus"].detach().cpu(),  # OPF format
                    "pred_forecast": pred_bus.detach().cpu(),  # NEW: Original format
                    "target_forecast": target_bus.detach().cpu(),  # NEW: Original
                    "mask_PQ": mask_PQ.cpu(),
                    "mask_PV": mask_PV.cpu(),
                    "mask_REF": mask_REF.cpu(),
                    "cost_predicted": cost_pred.detach().cpu(),
                    "cost_ground_truth": cost_gt.detach().cpu(),
                    "mean_residual_P_PQ": mean_res_P_PQ.detach().cpu(),
                    "max_residual_P_PQ": max_res_P_PQ.detach().cpu(),
                    "mean_residual_Q_PQ": mean_res_Q_PQ.detach().cpu(),
                    "max_residual_Q_PQ": max_res_Q_PQ.detach().cpu(),
                    "mean_residual_P_PV": mean_res_P_PV.detach().cpu(),
                    "max_residual_P_PV": max_res_P_PV.detach().cpu(),
                    "mean_residual_Q_PV": mean_res_Q_PV.detach().cpu(),
                    "max_residual_Q_PV": max_res_Q_PV.detach().cpu(),
                    "mean_residual_P_REF": mean_res_P_REF.detach().cpu(),
                    "max_residual_P_REF": max_res_P_REF.detach().cpu(),
                    "mean_residual_Q_REF": mean_res_Q_REF.detach().cpu(),
                    "max_residual_Q_REF": max_res_Q_REF.detach().cpu(),
                    "mask_Qg_violation": mask_Qg_violation.detach().cpu(),
                },
            )

        # Compute residuals (SAME as OPF)
        final_residual_real_bus = torch.mean(torch.abs(residual_P))
        final_residual_imag_bus = torch.mean(torch.abs(residual_Q))

        loss_dict["Active Power Loss"] = final_residual_real_bus.detach()
        loss_dict["Reactive Power Loss"] = final_residual_imag_bus.detach()

        # MSE per bus type (CHANGED: Use OPF format for consistency with plots)
        mse_PQ = F.mse_loss(
            output_opf["bus"][mask_PQ],
            target_opf["bus"][mask_PQ],
            reduction="none",
        )
        mse_PV = F.mse_loss(
            output_opf["bus"][mask_PV],
            target_opf["bus"][mask_PV],
            reduction="none",
        )
        mse_REF = F.mse_loss(
            output_opf["bus"][mask_REF],
            target_opf["bus"][mask_REF],
            reduction="none",
        )

        mse_PQ = mse_PQ.mean(dim=0)
        mse_PV = mse_PV.mean(dim=0)
        mse_REF = mse_REF.mean(dim=0)

        # Populate loss_dict (SAME as OPF)
        loss_dict["Opt gap"] = optimality_gap
        loss_dict["MSE PG"] = mse_PG[PG_H]

        loss_dict["Branch termal violation from"] = mean_thermal_violation_forward
        loss_dict["Branch termal violation to"] = mean_thermal_violation_reverse
        loss_dict["Branch voltage angle difference violations"] = (
            branch_angle_violation_mean
        )
        loss_dict["Mean Qg violation PV buses"] = mean_Qg_violation_PV
        loss_dict["Mean Qg violation REF buses"] = mean_Qg_violation_REF

        loss_dict["MSE PQ nodes - PG"] = mse_PQ[PG_OUT]
        loss_dict["MSE PV nodes - PG"] = mse_PV[PG_OUT]
        loss_dict["MSE REF nodes - PG"] = mse_REF[PG_OUT]

        loss_dict["MSE PQ nodes - QG"] = mse_PQ[QG_OUT]
        loss_dict["MSE PV nodes - QG"] = mse_PV[QG_OUT]
        loss_dict["MSE REF nodes - QG"] = mse_REF[QG_OUT]

        loss_dict["MSE PQ nodes - VM"] = mse_PQ[VM_OUT]
        loss_dict["MSE PV nodes - VM"] = mse_PV[VM_OUT]
        loss_dict["MSE REF nodes - VM"] = mse_REF[VM_OUT]

        loss_dict["MSE PQ nodes - VA"] = mse_PQ[VA_OUT]
        loss_dict["MSE PV nodes - VA"] = mse_PV[VA_OUT]
        loss_dict["MSE REF nodes - VA"] = mse_REF[VA_OUT]

        loss_dict["Test loss"] = loss_dict.pop("loss").detach()
        
        # Log all metrics (SAME as OPF)
        for metric, value in loss_dict.items():
            metric_name = f"{dataset_name}/{metric}"
            self.log(
                metric_name,
                value,
                batch_size=batch.num_graphs,
                add_dataloader_idx=False,
                sync_dist=True,
                logger=False,
            )
        return

    @rank_zero_only
    def on_test_end(self):
        """
        EXTENDED: Add forecast-specific CSV , then call parent.
        """
        # Get artifact directory (SAME as OPF)
        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir
        
        # Collect metrics (SAME as OPF)
        final_metrics = self.trainer.callback_metrics
        grouped_metrics = {}
        
        for full_key, value in final_metrics.items():
            try:
                value = value.item()
            except AttributeError:
                pass
            
            if "/" in full_key:
                dataset_name, metric = full_key.split("/", 1)
                if dataset_name not in grouped_metrics:
                    grouped_metrics[dataset_name] = {}
                grouped_metrics[dataset_name][metric] = value
        
        # Create forecast CSV
        for dataset, metrics in grouped_metrics.items():
            data_forecast = {
                "Feature": ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"],
                "MAE": [
                    metrics.get("MAE Pd (MW)", float("nan")),
                    metrics.get("MAE Qd (MVar)", float("nan")),
                    metrics.get("MAE Pg (MW)", float("nan")),
                    metrics.get("MAE Qg (MVar)", float("nan")),
                    metrics.get("MAE Vm (p.u.)", float("nan")),
                    metrics.get("MAE Va (rad)", float("nan")),
                ],
            }
            df_forecast = pd.DataFrame(data_forecast)
            
            test_dir = os.path.join(artifact_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            forecast_csv_path = os.path.join(test_dir, f"{dataset}_forecast_MAE.csv")
            df_forecast.to_csv(forecast_csv_path, index=False)
        
        
        #Call parent to generate OPF metrics/plots
        super().on_test_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """SAME: Not implemented (inherited from OPF)"""
        raise NotImplementedError
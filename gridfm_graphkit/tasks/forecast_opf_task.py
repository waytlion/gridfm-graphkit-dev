"""
ForecastOPF Task - One-step-ahead OPF forecasting (t → t+1)

Key differences from OptimalPowerFlowTask:
1. Predicts ALL dynamic features: [Pd, Qd, Qg, Vm, Va] (5D bus) + [Pg] (1D gen)
2. test_step() adds load forecast MAE metrics (Pd, Qd, Vm, Va, Qg, Pg)
3. Converts output to OPF format [Vm, Va, Pg, Qg] to reuse parent's physics/cost metrics

Inheritance: OptimalPowerFlowTask → ReconstructionTask → BaseTask
"""

from gridfm_graphkit.datasets.globals import (
    # Bus feature indices (used for format conversion)
    PD_H,
    QD_H,
    QG_H,
    VM_H,
    VA_H,
    # Generator feature indices
    PG_H,
    # Output feature indices (OPF format column order)
    VM_OUT,
    VA_OUT,
    PG_OUT,
    QG_OUT,
)
from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.io.registries import TASK_REGISTRY
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch_scatter import scatter_add
from lightning.pytorch.loggers import MLFlowLogger
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

    def training_step(self, batch):
        """
        Custom training step to log all loss components.
        
        The base ReconstructionTask only logs the total "loss".
        This override logs the individual components returned by the mixed loss.
        """
        _, loss_dict = self.shared_step(batch)
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        # Log total loss and LR
        self.log(
            "Training Loss",
            loss_dict["loss"].detach(),
            batch_size=batch.num_graphs,
            sync_dist=False,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            on_step=True,
        )
        self.log(
            "Learning Rate",
            current_lr,
            batch_size=batch.num_graphs,
            sync_dist=False,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            on_step=True,
        )
        
        # Log individual loss components natively returned by the MixedLoss
        for metric, value in loss_dict.items():
            if metric != "loss":
                self.log(
                    f"train/{metric}",
                    value.detach() if isinstance(value, torch.Tensor) else value,
                    batch_size=batch.num_graphs,
                    sync_dist=False,
                    on_epoch=True,  # Usually component losses are easier to read per epoch
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                )
                
        return loss_dict["loss"]

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
        
        # Aggregate generator Pg to buses (same logic as parent's test_step)
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
                pred_bus[:, VM_H],     # Vm
                pred_bus[:, VA_H],     # Va
                agg_pg.squeeze(),      # Pg aggregated from generators
                pred_bus[:, QG_H],     # Qg
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
                target_bus[:, VM_H],   # Vm
                target_bus[:, VA_H],   # Va
                agg_pg.squeeze(),      # Pg
                target_bus[:, QG_H],   # Qg
            ],
            dim=1,
        )
        
        return {"bus": target_opf, "gen": target_gen}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Forecast-specific test step.
        
        Structure:
        1. Get predictions and denormalize (same as OPF)
        2. Compute & log forecast-specific MAE metrics (Pd, Qd, Vm, Va, Qg, Pg)
        3. Convert output/target from forecast format to OPF format
        4. Delegate to parent's _compute_opf_metrics() for all physics metrics
        5. Log all metrics
        """
        #1: Prediction and Denormalization (SAME as OPF)
        output, loss_dict = self.shared_step(batch)
        dataset_name = self.args.data.networks[dataloader_idx]

        self.data_normalizers[dataloader_idx].inverse_transform(batch)
        self.data_normalizers[dataloader_idx].inverse_output(output, batch)

        # 2. Forecast specific MAE Metrics (capture how well the model predicts load & voltage at t+1.)
        pred_bus = output["bus"]        # [Pd, Qd, Qg, Vm, Va]
        target_bus = batch.y_dict["bus"]

        mae_metrics = {
            "MAE Pd (MW)":   (pred_bus[:, PD_H],  target_bus[:, PD_H]),
            "MAE Qd (MVar)": (pred_bus[:, QD_H],  target_bus[:, QD_H]),
            "MAE Qg (MVar)": (pred_bus[:, QG_H],  target_bus[:, QG_H]),
            "MAE Vm (p.u.)": (pred_bus[:, VM_H],  target_bus[:, VM_H]),
            "MAE Va (rad)":  (pred_bus[:, VA_H],  target_bus[:, VA_H]),
            "MAE Pg (MW)":   (output["gen"].squeeze(), batch.y_dict["gen"].squeeze()),
        }
        for name, (pred, tgt) in mae_metrics.items():
            self.log(
                f"{dataset_name}/{name}",
                torch.mean(torch.abs(pred - tgt)),
                batch_size=batch.num_graphs,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        # 3: Convert to OPF Format 
        output_opf = self._convert_to_opf_format(output, batch)
        target_opf = self._convert_target_to_opf_format(batch)

        # 4:  Reuse parent's OPF metrics (physics, cost, constraints) 
        opf_metrics = self._compute_opf_metrics(
            output_opf, target_opf, batch, dataset_name, dataloader_idx,
        )

        # 5: Log all test metrics
        # Use a separate dict instead of mutating loss_dict from shared_step().
        test_metrics = {**opf_metrics}
        test_metrics["Test loss"] = loss_dict.pop("loss").detach()
        for metric, value in test_metrics.items():
            self.log(
                f"{dataset_name}/{metric}",
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
        EXTENDED: Add forecast-specific CSV, then call parent.
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
        
        # Call parent to generate OPF metrics/plots
        super().on_test_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """SAME: Not implemented (inherited from OPF)"""
        raise NotImplementedError
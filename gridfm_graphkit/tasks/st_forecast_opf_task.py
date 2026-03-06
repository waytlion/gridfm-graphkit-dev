"""
ST-GNN Forecast OPF Task — PyTorch Lightning module for the
Space-then-Time multi-period AC-OPF forecaster.

Inherits from OptimalPowerFlowTask but overrides:
    - __init__: adds ST-specific losses (model loaded via MODELS_REGISTRY)
    - forward: unpacks collate_temporal dict
    - _shared_step / training_step / validation_step: dual-batch handling
    - test_step / on_test_end: horizon-wise evaluation and CSV output

The dataloader yields dicts from collate_temporal:
    {"folded_batch", "target_batch", "B", "W", "n", "N_bus"}
"""

import torch
from pytorch_lightning.utilities import rank_zero_only

from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.training.loss import ST_ForecastBusMSE, ST_ForecastGenMSE, ST_PhysicsForecastLoss
from gridfm_graphkit.datasets.globals import PD_H, QD_H, QG_H, VM_H, VA_H, PG_H


# Target feature indices for constructing the [B, N, n, 5] target tensor
BUS_TARGET_INDICES = [PD_H, QD_H, QG_H, VM_H, VA_H]  # -> [Pd, Qd, Qg, Vm, Va]
GEN_TARGET_INDICES = [PG_H]                             # -> [Pg]


@TASK_REGISTRY.register("ST_ForecastOPF")
class ST_ForecastOPFTask(OptimalPowerFlowTask):
    """
    PyTorch Lightning task for ST-GNN multi-period AC-OPF forecasting.

    Handles the dual-batch dataloader output from collate_temporal and
    composes weighted MSE (load + OPF) and physics forecast losses.
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

        # ---- ST-specific loss functions ----
        loss_args = _get_loss_args(args)

        self.mse_bus_loss_fn = ST_ForecastBusMSE(loss_args, args)
        self.mse_gen_loss_fn = ST_ForecastGenMSE(loss_args, args)
        self.physics_loss_fn = ST_PhysicsForecastLoss(loss_args, args)

        # Loss weights
        self.lambda_mse = getattr(loss_args, "lambda_mse", 1.0)
        self.lambda_gen = getattr(loss_args, "lambda_gen", 0.1)
        self.lambda_phys = getattr(loss_args, "lambda_phys", 0.1)

        self.horizon_metrics = {i: [] for i in range(len(args.data.networks))}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch):
        """Unpack collate_temporal dict and call ST-GNN model."""
        folded_batch = batch["folded_batch"]
        target_batch = batch["target_batch"]
        B = batch["B"]
        W = batch["W"]
        N_bus = batch["N_bus"]
        return self.model(folded_batch, target_batch, B, W, N_bus)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(self, batch, prefix="train"):
        """
        Forward pass + loss computation for train / val.

        batch: dict from collate_temporal with keys:
            folded_batch, target_batch, B, W, n, N_bus
        prefix: "train" or "val" for metric key naming

        Returns (total_loss, log_dict)
        """
        pred = self(batch)

        # --- Construct target dict [B, N_bus, n, F] from target_batch ---
        target_batch = batch["target_batch"]
        B = batch["B"]
        n = batch["n"]
        N_bus = batch["N_bus"]

        # target_batch["bus"].x: [B*n*N_bus, 15] -> [B, n, N_bus, 15] -> [B, N_bus, n, 15]
        target_bus_x = target_batch["bus"].x
        target_bus_4d = target_bus_x.view(B, n, N_bus, -1).permute(0, 2, 1, 3)
        target_bus = target_bus_4d[..., BUS_TARGET_INDICES]  # [B, N_bus, n, 5]

        target_gen_x = target_batch["gen"].x
        N_gen = target_gen_x.size(0) // (B * n)
        target_gen_4d = target_gen_x.view(B, n, N_gen, -1).permute(0, 2, 1, 3)
        target_gen = target_gen_4d[..., GEN_TARGET_INDICES]  # [B, N_gen, n, 1]

        target = {"bus": target_bus, "gen": target_gen}

        # --- MSE losses ---
        mse_bus_out = self.mse_bus_loss_fn(pred, target)
        mse_gen_out = self.mse_gen_loss_fn(pred, target)

        # --- Physics loss ---
        mask = {
            "B": B,
            "n": n,
            "N_bus": N_bus,
            "bus_x": target_batch["bus"].x,  # [B*n*N_bus, F] for physics decoder
        }
        phys_out = self.physics_loss_fn(
            pred, target,
            target_batch.edge_index_dict,
            target_batch.edge_attr_dict,
            mask,
        )

        # --- Weighted total ---
        total_loss = (
            self.lambda_mse * mse_bus_out["loss"]
            + self.lambda_gen * mse_gen_out["loss"]
            + self.lambda_phys * phys_out["loss"]
        )

        # --- Logging dict ---
        log_dict = {f"{prefix}/total_loss": total_loss.detach()}
        for key, val in mse_bus_out.items():
            if key != "loss":
                log_dict[f"{prefix}/{key}"] = val.detach() if isinstance(val, torch.Tensor) else val
        for key, val in mse_gen_out.items():
            if key != "loss":
                log_dict[f"{prefix}/{key}"] = val.detach() if isinstance(val, torch.Tensor) else val
        for key, val in phys_out.items():
            if key != "loss":
                log_dict[f"{prefix}/{key}"] = val.detach() if isinstance(val, torch.Tensor) else val

        return total_loss, log_dict, pred

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        total_loss, log_dict, _ = self._shared_step(batch, prefix="train")

        # Log LR
        current_lr = self.optimizers().param_groups[0]["lr"]
        log_dict["Learning Rate"] = current_lr

        self.log_dict(
            log_dict,
            batch_size=batch["B"],
            sync_dist=False,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        total_loss, log_dict, _ = self._shared_step(batch, prefix="val")

        # Lightning needs "Validation loss" for ReduceLROnPlateau monitor
        log_dict["Validation loss"] = total_loss.detach()

        self.log_dict(
            log_dict,
            batch_size=batch["B"],
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        return total_loss

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        from torch_scatter import scatter_add

        # 1. Forward pass
        total_loss, log_dict, pred = self._shared_step(batch, prefix="test")
        dataset_name = self.args.data.networks[dataloader_idx]

        target_batch = batch["target_batch"]
        B = batch["B"]
        n = batch["n"]
        N_bus = batch["N_bus"]
        N_gen = target_batch["gen"].x.size(0) // (B * n)

        # 2. Flatten predictions to [B*n*N, F] for denormalization
        pred_bus_flat = pred["bus"].permute(0, 2, 1, 3).reshape(B * n * N_bus, 5)
        pred_gen_flat = pred["gen"].permute(0, 2, 1, 3).reshape(B * n * N_gen, 1)
        flat_pred = {"bus": pred_bus_flat, "gen": pred_gen_flat}

        # 3. Denormalize
        self.data_normalizers[dataloader_idx].inverse_transform(target_batch)
        self.data_normalizers[dataloader_idx].inverse_output(flat_pred, target_batch)

        # 4. Reconstruct 4D targets from denormalized target_batch [B, N_bus, n, F]
        target_bus_4d = target_batch["bus"].x.view(B, n, N_bus, -1).permute(0, 2, 1, 3)
        target_bus = target_bus_4d[..., BUS_TARGET_INDICES]  # [B, N_bus, n, 5]
        target_gen_4d = target_batch["gen"].x.view(B, n, N_gen, -1).permute(0, 2, 1, 3)
        target_gen = target_gen_4d[..., GEN_TARGET_INDICES]  # [B, N_gen, n, 1]

        # Reshape flat_pred back to [B, N_bus, n, F] for MAE
        pred_bus = flat_pred["bus"].view(B, n, N_bus, 5).permute(0, 2, 1, 3)  # [B, N_bus, n, 5]
        pred_gen = flat_pred["gen"].view(B, n, N_gen, 1).permute(0, 2, 1, 3)  # [B, N_gen, n, 1]

        # 5. Scalar MAE metrics (mean over all dims)
        mae_metrics = {
            "MAE Pd (MW)":   (pred_bus[..., 0], target_bus[..., 0]),
            "MAE Qd (MVar)": (pred_bus[..., 1], target_bus[..., 1]),
            "MAE Qg (MVar)": (pred_bus[..., 2], target_bus[..., 2]),
            "MAE Vm (p.u.)": (pred_bus[..., 3], target_bus[..., 3]),
            "MAE Va (rad)":  (pred_bus[..., 4], target_bus[..., 4]),
            "MAE Pg (MW)":   (pred_gen[..., 0], target_gen[..., 0]),
        }
        for name, (p, tgt) in mae_metrics.items():
            self.log(
                f"{dataset_name}/{name}",
                torch.mean(torch.abs(p - tgt)),
                batch_size=B,
                add_dataloader_idx=False,
                sync_dist=True,
            )

        # 6. Horizon-wise MAE: mean over (B, N_bus) dims -> [n, F]
        # pred_bus shape: [B, N_bus, n, 5]  dims: 0=B, 1=N_bus, 2=n, 3=F
        abs_err_bus = torch.abs(pred_bus - target_bus)   # [B, N_bus, n, 5]
        abs_err_gen = torch.abs(pred_gen - target_gen)   # [B, N_gen, n, 1]
        horizon_mae_bus = abs_err_bus.mean(dim=(0, 1))   # [n, 5]
        horizon_mae_gen = abs_err_gen.mean(dim=(0, 1))   # [n, 1]
        # Store as [n, 6] tensor with columns [Pd, Qd, Qg, Vm, Va, Pg]
        horizon_mae = torch.cat([horizon_mae_bus, horizon_mae_gen], dim=-1)  # [n, 6]
        self.horizon_metrics[dataloader_idx].append(horizon_mae.detach().cpu())

        # 7. Build OPF-format tensors [B*n*N_bus, 4] for _compute_opf_metrics
        _, gen_to_bus_index = target_batch.edge_index_dict[("gen", "connected_to", "bus")]
        agg_pg = scatter_add(
            flat_pred["gen"],
            gen_to_bus_index,
            dim=0,
            dim_size=B * n * N_bus,
        )
        output_opf = {
            "bus": torch.stack(
                [
                    flat_pred["bus"][:, 3],  # Vm
                    flat_pred["bus"][:, 4],  # Va
                    agg_pg.squeeze(),         # Pg aggregated
                    flat_pred["bus"][:, 2],  # Qg
                ],
                dim=1,
            ),
            "gen": flat_pred["gen"],
        }
        _, gen_to_bus_index_tgt = target_batch.edge_index_dict[("gen", "connected_to", "bus")]
        agg_pg_tgt = scatter_add(
            target_gen_4d.permute(0, 2, 1, 3).reshape(B * n * N_gen, 1),
            gen_to_bus_index_tgt,
            dim=0,
            dim_size=B * n * N_bus,
        )
        target_opf = {
            "bus": torch.stack(
                [
                    target_bus[..., 3].permute(0, 2, 1).reshape(B * n * N_bus),  # Vm
                    target_bus[..., 4].permute(0, 2, 1).reshape(B * n * N_bus),  # Va
                    agg_pg_tgt.squeeze(),                                          # Pg
                    target_bus[..., 2].permute(0, 2, 1).reshape(B * n * N_bus),  # Qg
                ],
                dim=1,
            ),
            "gen": target_gen_4d.permute(0, 2, 1, 3).reshape(B * n * N_gen, 1),
        }

        # 8. Delegate OPF physics/cost/constraint metrics to parent
        opf_metrics = self._compute_opf_metrics(
            output_opf, target_opf, target_batch, dataset_name, dataloader_idx,
        )

        # 9. Log all test metrics
        test_metrics = {**opf_metrics}
        test_metrics["Test loss"] = total_loss.detach()
        for metric, value in test_metrics.items():
            self.log(
                f"{dataset_name}/{metric}",
                value,
                batch_size=B,
                add_dataloader_idx=False,
                sync_dist=True,
                logger=False,
            )
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    @rank_zero_only
    def on_test_end(self):
        import os
        import pandas as pd
        from lightning.pytorch.loggers import MLFlowLogger

        # Get artifact directory (same logic as parent)
        if isinstance(self.logger, MLFlowLogger):
            artifact_dir = os.path.join(
                self.logger.save_dir,
                self.logger.experiment_id,
                self.logger.run_id,
                "artifacts",
            )
        else:
            artifact_dir = self.logger.save_dir

        test_dir = os.path.join(artifact_dir, "test")
        os.makedirs(test_dir, exist_ok=True)

        # Write horizon-wise MAE CSV per dataset
        # Columns: [Pd, Qd, Qg, Vm, Va, Pg], index: timestep t (1..n)
        for dataset_idx, batch_maes in self.horizon_metrics.items():
            if not batch_maes:
                continue
            dataset_name = self.args.data.networks[dataset_idx]
            # Average across all test batches: each entry is [n, 6]
            horizon_mae = torch.stack(batch_maes, dim=0).mean(dim=0)  # [n, 6]
            df_horizon = pd.DataFrame(
                horizon_mae.numpy(),
                columns=["Pd (MW)", "Qd (MVar)", "Qg (MVar)", "Vm (p.u.)", "Va (rad)", "Pg (MW)"],
                index=pd.RangeIndex(start=1, stop=horizon_mae.size(0) + 1, name="t"),
            )
            horizon_csv_path = os.path.join(test_dir, f"{dataset_name}_horizon_MAE.csv")
            df_horizon.to_csv(horizon_csv_path)

        self.horizon_metrics.clear()

        # Delegate forecast_MAE CSV + RMSE.csv + metrics.csv to parent chain
        super().on_test_end()


# ======================================================================
# Helper
# ======================================================================

class _LossArgNamespace:
    """Minimal namespace to pass loss_args to loss constructors."""
    pass


def _get_loss_args(args):
    """Extract ST loss config from args, with defaults."""
    ns = _LossArgNamespace()
    training = getattr(args, "training", None)
    st_loss = None
    if training is not None:
        st_loss = getattr(training, "st_loss", None)

    ns.lambda_load = getattr(st_loss, "lambda_load", 0.5) if st_loss else 0.5
    ns.lambda_opf = getattr(st_loss, "lambda_opf", 0.5) if st_loss else 0.5
    ns.lambda_mse = getattr(st_loss, "lambda_mse", 1.0) if st_loss else 1.0
    ns.lambda_gen = getattr(st_loss, "lambda_gen", 0.1) if st_loss else 0.1
    ns.lambda_phys = getattr(st_loss, "lambda_phys", 0.1) if st_loss else 0.1
    return ns

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
        # Collect raw tensors per batch for proper global metric computation
        self.forecast_preds = {i: [] for i in range(len(args.data.networks))}
        self.forecast_targets = {i: [] for i in range(len(args.data.networks))}
        self.forecast_naive = {i: [] for i in range(len(args.data.networks))}

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

        mask = {
            "B": B,
            "n": n,
            "N_bus": N_bus,
            "bus_x": target_batch["bus"].x,
        }

        loss_dict = self.loss_fn(
            pred, target,
            target_batch.edge_index_dict,
            target_batch.edge_attr_dict,
            mask,
        )
        total_loss = loss_dict.pop("loss")

        log_dict = {f"{prefix}/total_loss": total_loss.detach()}
        for key, val in loss_dict.items():
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

        folded_batch = batch["folded_batch"]
        target_batch = batch["target_batch"]
        B = batch["B"]
        W = batch["W"]
        n = batch["n"]
        N_bus = batch["N_bus"]
        N_gen = target_batch["gen"].x.size(0) // (B * n)

        # 2. Flatten predictions to [B*n*N, F] for denormalization
        pred_bus_flat = pred["bus"].permute(0, 2, 1, 3).reshape(B * n * N_bus, 5)
        pred_gen_flat = pred["gen"].permute(0, 2, 1, 3).reshape(B * n * N_gen, 1)
        flat_pred = {"bus": pred_bus_flat, "gen": pred_gen_flat}

        # 3. Denormalize predictions and targets
        self.data_normalizers[dataloader_idx].inverse_transform(target_batch)
        self.data_normalizers[dataloader_idx].inverse_output(flat_pred, target_batch)

        # 3b. Denormalize the lookback window (needed for naive baseline)
        self.data_normalizers[dataloader_idx].inverse_transform(folded_batch)

        # 4. Reconstruct 4D targets from denormalized target_batch [B, N_bus, n, F]
        target_bus_4d = target_batch["bus"].x.view(B, n, N_bus, -1).permute(0, 2, 1, 3)
        target_bus = target_bus_4d[..., BUS_TARGET_INDICES]  # [B, N_bus, n, 5]
        target_gen_4d = target_batch["gen"].x.view(B, n, N_gen, -1).permute(0, 2, 1, 3)
        target_gen = target_gen_4d[..., GEN_TARGET_INDICES]  # [B, N_gen, n, 1]

        # Reshape flat_pred back to [B, N_bus, n, F] for metrics
        pred_bus = flat_pred["bus"].view(B, n, N_bus, 5).permute(0, 2, 1, 3)  # [B, N_bus, n, 5]
        pred_gen = flat_pred["gen"].view(B, n, N_gen, 1).permute(0, 2, 1, 3)  # [B, N_gen, n, 1]

        # 5. Build naive baseline: value from 48 hours ago (or earliest available) repeated for all n horizons
        #    folded_batch is B*W graphs ordered sample-major, time-minor.
        #    Using 48-hour lag as naive baseline when W >= 48, otherwise use earliest window timestep
        window_bus_4d = folded_batch["bus"].x.view(B, W, N_bus, -1)  # [B, W, N_bus, F_full]
        lag_idx = min(48, W)  # Use 48 hours ago if available, else earliest timestep
        lag_bus = window_bus_4d[:, -lag_idx, :, :]  # [B, N_bus, F_full]
        naive_bus = lag_bus[:, :, BUS_TARGET_INDICES].unsqueeze(2).expand_as(pred_bus)  # [B, N_bus, n, 5]

        N_gen_window = folded_batch["gen"].x.size(0) // (B * W)
        window_gen_4d = folded_batch["gen"].x.view(B, W, N_gen_window, -1)
        lag_gen = window_gen_4d[:, -lag_idx, :, :]  # [B, N_gen, F_gen_full]
        naive_gen = lag_gen[:, :, GEN_TARGET_INDICES].unsqueeze(2).expand_as(pred_gen)  # [B, N_gen, n, 1]

        # 6. Store raw tensors (detached, on CPU) for global metrics in on_test_end
        self.forecast_preds[dataloader_idx].append({
            "bus": pred_bus.detach().cpu(),   # [B, N_bus, n, 5]
            "gen": pred_gen.detach().cpu(),   # [B, N_gen, n, 1]
        })
        self.forecast_targets[dataloader_idx].append({
            "bus": target_bus.detach().cpu(),
            "gen": target_gen.detach().cpu(),
        })
        self.forecast_naive[dataloader_idx].append({
            "bus": naive_bus.detach().cpu(),
            "gen": naive_gen.detach().cpu(),
        })

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
            target_gen.permute(0, 2, 1, 3).reshape(B * n * N_gen, 1),
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
            "gen": target_gen.permute(0, 2, 1, 3).reshape(B * n * N_gen, 1),
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
                logger=True,
            )
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    # ------------------------------------------------------------------
    # Forecast metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_forecast_metrics(pred, target, naive):
        """
        Compute per-timestep and global forecast metrics for a single variable.

        All inputs are [total_samples, N_nodes, n] (spatial dim already present).

        Returns dict of 1-D tensors (length n) for per-step metrics
        and scalars for global metrics.

        Metrics
        -------
        RMSE  : sqrt(mean( (pred - target)^2 ))               per timestep
        MAE   : mean( |pred - target| )                       per timestep
        wMAPE : sum(|pred - target|) / (sum(|target|) + eps)   per timestep
                (weighted MAPE avoids per-element div-by-zero)
        MASE  : MAE_model / (MAE_naive + eps)                  per timestep
        MSSE  : MSE_model / (MSE_naive + eps)                  per timestep

        Naive baseline: 48-hour lag when available (W >= 48), otherwise earliest window timestep
        """
        eps = 1e-8

        err = pred - target                     # [S, N, n]
        abs_err = torch.abs(err)                # [S, N, n]
        sq_err = err ** 2                       # [S, N, n]

        naive_err = naive - target
        naive_abs_err = torch.abs(naive_err)
        naive_sq_err = naive_err ** 2

        # --- per-timestep (mean over samples & nodes, keep n) ---
        mae_t = abs_err.mean(dim=(0, 1))                          # [n]
        rmse_t = sq_err.mean(dim=(0, 1)).sqrt()                   # [n]
        wmape_t = abs_err.sum(dim=(0, 1)) / (torch.abs(target).sum(dim=(0, 1)) + eps)  # [n]
        naive_mae_t = naive_abs_err.mean(dim=(0, 1))              # [n]
        naive_mse_t = naive_sq_err.mean(dim=(0, 1))               # [n]
        mase_t = mae_t / (naive_mae_t + eps)                      # [n]
        msse_t = sq_err.mean(dim=(0, 1)) / (naive_mse_t + eps)    # [n]

        # --- global (mean over everything) ---
        mae_g = abs_err.mean()
        rmse_g = sq_err.mean().sqrt()
        wmape_g = abs_err.sum() / (torch.abs(target).sum() + eps)
        naive_mae_g = naive_abs_err.mean()
        naive_mse_g = naive_sq_err.mean()
        mase_g = mae_g / (naive_mae_g + eps)
        msse_g = sq_err.mean() / (naive_mse_g + eps)

        return {
            "rmse_t": rmse_t, "mae_t": mae_t, "wmape_t": wmape_t,
            "mase_t": mase_t, "msse_t": msse_t,
            "rmse": rmse_g.item(), "mae": mae_g.item(), "wmape": wmape_g.item(),
            "mase": mase_g.item(), "msse": msse_g.item(),
        }

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

        VAR_NAMES = ["Pd (MW)", "Qd (MVar)", "Qg (MVar)", "Vm (p.u.)", "Va (rad)", "Pg (MW)"]
        METRIC_NAMES = ["RMSE", "MAE", "wMAPE", "MASE", "MSSE"]

        for dataset_idx in range(len(self.args.data.networks)):
            if not self.forecast_preds[dataset_idx]:
                continue
            dataset_name = self.args.data.networks[dataset_idx]

            # Concatenate all batches along sample dim (dim 0)
            all_pred_bus = torch.cat([d["bus"] for d in self.forecast_preds[dataset_idx]], dim=0)    # [S, N_bus, n, 5]
            all_pred_gen = torch.cat([d["gen"] for d in self.forecast_preds[dataset_idx]], dim=0)    # [S, N_gen, n, 1]
            all_tgt_bus = torch.cat([d["bus"] for d in self.forecast_targets[dataset_idx]], dim=0)
            all_tgt_gen = torch.cat([d["gen"] for d in self.forecast_targets[dataset_idx]], dim=0)
            all_naive_bus = torch.cat([d["bus"] for d in self.forecast_naive[dataset_idx]], dim=0)
            all_naive_gen = torch.cat([d["gen"] for d in self.forecast_naive[dataset_idx]], dim=0)

            # Compute metrics per variable.  Shapes are [S, N, n] after indexing the feature dim.
            var_metrics = []
            for feat_idx in range(5):  # bus features: Pd, Qd, Qg, Vm, Va
                m = self._compute_forecast_metrics(
                    all_pred_bus[..., feat_idx],
                    all_tgt_bus[..., feat_idx],
                    all_naive_bus[..., feat_idx],
                )
                var_metrics.append(m)
            # Pg (generator)
            m_pg = self._compute_forecast_metrics(
                all_pred_gen[..., 0],
                all_tgt_gen[..., 0],
                all_naive_gen[..., 0],
            )
            var_metrics.append(m_pg)

            n_horizon = all_pred_bus.size(2)

            # ── Build per-timestep CSV ──────────────────────────────────
            # Layout: rows = timestep (1..n), then a "GLOBAL" summary row
            # Columns: single-level flattened format "Variable - Metric"
            rows_per_t = []
            for t in range(n_horizon):
                row = {}
                for vi, vname in enumerate(VAR_NAMES):
                    vm = var_metrics[vi]
                    row[f"{vname} - RMSE"] = vm["rmse_t"][t].item()
                    row[f"{vname} - MAE"] = vm["mae_t"][t].item()
                    row[f"{vname} - wMAPE"] = vm["wmape_t"][t].item()
                    row[f"{vname} - MASE"] = vm["mase_t"][t].item()
                    row[f"{vname} - MSSE"] = vm["msse_t"][t].item()
                rows_per_t.append(row)

            # Global summary row
            global_row = {}
            for vi, vname in enumerate(VAR_NAMES):
                vm = var_metrics[vi]
                global_row[f"{vname} - RMSE"] = vm["rmse"]
                global_row[f"{vname} - MAE"] = vm["mae"]
                global_row[f"{vname} - wMAPE"] = vm["wmape"]
                global_row[f"{vname} - MASE"] = vm["mase"]
                global_row[f"{vname} - MSSE"] = vm["msse"]

            index_labels = [f"t+{t+1}" for t in range(n_horizon)] + ["GLOBAL"]
            df = pd.DataFrame(rows_per_t + [global_row], index=index_labels)
            df.index.name = "Horizon"

            forecast_csv_path = os.path.join(test_dir, f"{dataset_name}_forecast.csv")
            df.to_csv(forecast_csv_path)

        # Cleanup
        self.forecast_preds = {i: [] for i in range(len(self.args.data.networks))}
        self.forecast_targets = {i: [] for i in range(len(self.args.data.networks))}
        self.forecast_naive = {i: [] for i in range(len(self.args.data.networks))}

        # Delegate RMSE.csv + metrics.csv (OPF physics metrics) to parent
        super().on_test_end()




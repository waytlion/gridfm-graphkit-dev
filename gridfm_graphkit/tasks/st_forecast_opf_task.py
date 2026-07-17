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
from torch_geometric.data import Batch
from pytorch_lightning.utilities import rank_zero_only

from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.opf_task import OptimalPowerFlowTask
from gridfm_graphkit.models.utils import ComputeBranchFlow
from gridfm_graphkit.tasks.constraint_metrics import (
    ConstraintViolationAccumulator,
    per_type_from_opf_batch,
    canonical_scalar_rows,
)
from gridfm_graphkit.datasets.globals import (
    PD_H, QD_H, QG_H, VM_H, VA_H, PG_H, PQ_H, PV_H,
    C0_H, C1_H, C2_H,
)


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
        self.forecast_gaps = {i: [] for i in range(len(args.data.networks))}
        self.forecast_mode = getattr(args.data, "forecast_mode", "direct")
        # Bus metadata for diagnostic plots (populated on first test batch)
        self._bus_type_pq = None
        self._bus_type_pv = None
        self._gen_to_bus = None
        self._N_bus = None

        # Timing and sample counters
        self.train_time_total_s = 0.0
        self.train_samples = 0
        self.total_test_sequences = 0

        # Canonical constraint-violation accumulators (per network)
        self._branch_flow = ComputeBranchFlow()
        self._viol_acc = {i: ConstraintViolationAccumulator() for i in range(len(args.data.networks))}

    def on_train_batch_start(self, batch, batch_idx):
        import time
        self._train_batch_start_time = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - self._train_batch_start_time
        self.train_time_total_s += duration
        self.train_samples += int(batch["B"])

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

    def _select_target_step(self, target_batch, B, n, step_index):
        target_graphs = target_batch.to_data_list()
        step_graphs = [target_graphs[i * n + step_index] for i in range(B)]
        return Batch.from_data_list(step_graphs)

    def _autoregressive_rollout(self, batch, dataloader_idx=0):
        folded_batch = batch["folded_batch"]
        target_batch = batch["target_batch"]
        B = batch["B"]
        W = batch["W"]
        n = batch["n"]
        N_bus = batch["N_bus"]

        window_graphs = folded_batch.to_data_list()
        target_graphs = target_batch.to_data_list()

        windows = [window_graphs[i * W : (i + 1) * W] for i in range(B)]
        targets = [target_graphs[i * n : (i + 1) * n] for i in range(B)]

        pred_bus_steps = []
        pred_gen_steps = []

        with torch.no_grad():
            for step in range(n):
                all_window_graphs = []
                target_step_graphs = []
                for i in range(B):
                    all_window_graphs.extend(windows[i])
                    target_step_graphs.append(targets[i][step])

                folded_step = Batch.from_data_list(all_window_graphs)
                target_step = Batch.from_data_list(target_step_graphs)

                if self._time_forward:
                    import time
                    _cuda = torch.cuda.is_available()
                    if _cuda:
                        torch.cuda.synchronize()
                    _t0 = time.perf_counter()

                pred_step = self.model(
                    folded_step,
                    target_step,
                    B,
                    W,
                    N_bus,
                    one_step=True,
                    step_index=0,
                )

                if self._time_forward:
                    if _cuda:
                        torch.cuda.synchronize()
                    self._infer_time_s += time.perf_counter() - _t0
                    self._infer_samples += int(B)

                if isinstance(pred_step, tuple):
                    pred_step = {"bus": pred_step[0], "gen": pred_step[1]}

                pred_bus_steps.append(pred_step["bus"])
                pred_gen_steps.append(pred_step["gen"])

                pred_bus = pred_step["bus"][:, :, 0, :]
                pred_gen = pred_step["gen"][:, :, 0, :]

                for i in range(B):
                    next_graph = target_step_graphs[i].clone()

                    bus_x = next_graph["bus"].x
                    bus_x[:, PD_H] = pred_bus[i, :, 0]
                    bus_x[:, QD_H] = pred_bus[i, :, 1]
                    bus_x[:, QG_H] = pred_bus[i, :, 2]
                    bus_x[:, VM_H] = pred_bus[i, :, 3]
                    bus_x[:, VA_H] = pred_bus[i, :, 4]
                    next_graph["bus"].x = bus_x

                    gen_x = next_graph["gen"].x
                    gen_x[:, PG_H] = pred_gen[i, :, 0]
                    next_graph["gen"].x = gen_x

                    windows[i] = windows[i][1:] + [next_graph]

        pred_bus = torch.cat(pred_bus_steps, dim=2)  # [B, N_bus, n, 5]
        pred_gen = torch.cat(pred_gen_steps, dim=2)  # [B, N_gen, n, 1]
        return {"bus": pred_bus, "gen": pred_gen}

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(self, batch, prefix="train", dataloader_idx=0):
        """
        Forward pass + loss computation for train / val.

        batch: dict from collate_temporal with keys:
            folded_batch, target_batch, B, W, n, N_bus
        prefix: "train" or "val" for metric key naming

        Returns (total_loss, log_dict)
        """
        folded_batch = batch["folded_batch"]
        target_batch = batch["target_batch"]
        B = batch["B"]
        W = batch["W"]
        n = batch["n"]
        N_bus = batch["N_bus"]

        use_one_step = self.forecast_mode == "autoregressive" and prefix in {"train", "val"}

        if self._time_forward:
            import time  # torch is already imported at module scope
            _cuda = torch.cuda.is_available()
            if _cuda:
                torch.cuda.synchronize()
            _t0 = time.perf_counter()

        if use_one_step:
            target_batch = self._select_target_step(target_batch, B, n, 0)
            pred = self.model(folded_batch, target_batch, B, W, N_bus, one_step=True, step_index=0)
            n = 1
        else:
            pred = self.model(folded_batch, target_batch, B, W, N_bus)

        if self._time_forward:
            if _cuda:
                torch.cuda.synchronize()
            self._infer_time_s += time.perf_counter() - _t0
            self._infer_samples += int(B * n)  # B windows x n horizon steps = dispatch decisions
        if isinstance(pred, tuple):
            pred = {"bus": pred[0], "gen": pred[1]}
        # --- Construct target dict [B, N_bus, n, F] from target_batch ---

        # target_batch["bus"].x: [B*n*N_bus, 15] -> [B, n, N_bus, 15] -> [B, N_bus, n, 15]
        target_bus_x = target_batch["bus"].x
        target_bus_4d = target_bus_x.view(B, n, N_bus, -1).permute(0, 2, 1, 3)
        target_bus = target_bus_4d[..., BUS_TARGET_INDICES]  # [B, N_bus, n, 5]

        target_gen_x = target_batch["gen"].x
        N_gen = target_gen_x.size(0) // (B * n)
        target_gen_4d = target_gen_x.view(B, n, N_gen, -1).permute(0, 2, 1, 3)
        target_gen = target_gen_4d[..., GEN_TARGET_INDICES]  # [B, N_gen, n, 1]

        target = {"bus": target_bus, "gen": target_gen}

        static_baseMVA = None
        if hasattr(self, "data_normalizers") and len(self.data_normalizers) > dataloader_idx:
            norm = self.data_normalizers[dataloader_idx]
            static_baseMVA = getattr(norm, "baseMVA_static", getattr(norm, "baseMVA", None))


        mask = {
            "B": B,
            "n": n,
            "N_bus": N_bus,
            "bus_x": target_batch["bus"].x,
            "gen_x_4d": target_gen_4d,
            "window_baseMVA": batch.get("window_baseMVA", None),
            "static_baseMVA": static_baseMVA,
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

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        total_loss, log_dict, _ = self._shared_step(batch, prefix="train", dataloader_idx=dataloader_idx)

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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        total_loss, log_dict, _ = self._shared_step(batch, prefix="val", dataloader_idx=dataloader_idx)

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

        self.total_test_sequences += int(batch["B"])

        # 1. Forward pass
        if self.forecast_mode == "autoregressive":
            pred = self._autoregressive_rollout(batch, dataloader_idx=dataloader_idx)
            if isinstance(pred, tuple):
                pred = {"bus": pred[0], "gen": pred[1]}

            target_batch = batch["target_batch"]
            B = batch["B"]
            n = batch["n"]
            N_bus = batch["N_bus"]

            target_bus_x = target_batch["bus"].x
            target_bus_4d = target_bus_x.view(B, n, N_bus, -1).permute(0, 2, 1, 3)
            target_bus = target_bus_4d[..., BUS_TARGET_INDICES]

            target_gen_x = target_batch["gen"].x
            N_gen = target_gen_x.size(0) // (B * n)
            target_gen_4d = target_gen_x.view(B, n, N_gen, -1).permute(0, 2, 1, 3)
            target_gen = target_gen_4d[..., GEN_TARGET_INDICES]

            target = {"bus": target_bus, "gen": target_gen}

            static_baseMVA = None
            if hasattr(self, "data_normalizers") and len(self.data_normalizers) > dataloader_idx:
                norm = self.data_normalizers[dataloader_idx]
                static_baseMVA = getattr(norm, "baseMVA_static", getattr(norm, "baseMVA", None))

            mask = {
                "B": B,
                "n": n,
                "N_bus": N_bus,
                "bus_x": target_batch["bus"].x,
                "gen_x_4d": target_gen_4d,
                "window_baseMVA": batch.get("window_baseMVA", None),
                "static_baseMVA": static_baseMVA,
            }

            loss_dict = self.loss_fn(
                pred,
                target,
                target_batch.edge_index_dict,
                target_batch.edge_attr_dict,
                mask,
            )
            total_loss = loss_dict.pop("loss")
            log_dict = {f"test/total_loss": total_loss.detach()}
        else:
            total_loss, log_dict, pred = self._shared_step(batch, prefix="test", dataloader_idx=dataloader_idx)
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
        from gridfm_graphkit.datasets.normalizers import HeteroDataWindowMVANormalizer
        _normalizer = self.data_normalizers[dataloader_idx]
        _is_window_norm = isinstance(_normalizer, HeteroDataWindowMVANormalizer)
        _window_baseMVA = batch.get("window_baseMVA", None)

        if _is_window_norm:
            _normalizer.inverse_transform(target_batch, _window_baseMVA)
            _normalizer.inverse_output(flat_pred, target_batch, _window_baseMVA)
        else:
            _normalizer.inverse_transform(target_batch)
            _normalizer.inverse_output(flat_pred, target_batch)

        # 3b. Denormalize the lookback window (needed for naive baseline)
        if _is_window_norm:
            _normalizer.inverse_transform(folded_batch, _window_baseMVA)
        else:
            _normalizer.inverse_transform(folded_batch)

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

        # 6a. Compute optimality gap per horizon step [%]
        c0 = target_gen_4d[..., C0_H]
        c1 = target_gen_4d[..., C1_H]
        c2 = target_gen_4d[..., C2_H]
        pg_p = pred_gen.squeeze(-1)
        pg_t = target_gen.squeeze(-1)

        _cost_p = (c0 + c1 * pg_p + c2 * pg_p**2).sum(dim=1)  # [B, n]
        _cost_t = (c0 + c1 * pg_t + c2 * pg_t**2).sum(dim=1)  # [B, n]
        _gaps = torch.abs(_cost_p - _cost_t) / (_cost_t + 1e-8) * 100
        self.forecast_gaps[dataloader_idx].append(_gaps.detach().cpu())

        # 6b. Store bus type masks and gen-to-bus mapping (static, first batch only)
        if self._bus_type_pq is None:
            target_bus_full = target_batch["bus"].x.view(B, n, N_bus, -1)
            self._bus_type_pq = (target_bus_full[0, 0, :, PQ_H] == 1).cpu()  # [N_bus]
            self._bus_type_pv = (target_bus_full[0, 0, :, PV_H] == 1).cpu()  # [N_bus]
            _, g2b = target_batch.edge_index_dict[("gen", "connected_to", "bus")]
            self._gen_to_bus = (g2b[:N_gen] % N_bus).cpu()  # [N_gen] within single graph
            self._N_bus = N_bus

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

        # 8b. Accumulate canonical constraint-violation stats (same extraction as
        # the surrogate arm). Each (window, horizon-step) graph in target_batch is
        # one scenario — matching compare.py flattening horizon into scenarios.
        per_type, num_scen = per_type_from_opf_batch(output_opf, target_batch, self._branch_flow)
        self._viol_acc[dataloader_idx].update(per_type, num_scen)

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
                reduce_fx="max" if "viol. max" in metric else "mean",
            )
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.forecast_mode == "autoregressive":
            return self._autoregressive_rollout(batch, dataloader_idx=dataloader_idx)
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

        custom_csv_data = {}

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
            all_gaps = torch.cat(self.forecast_gaps[dataset_idx], dim=0)    # [S, n]

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
            gap_t = all_gaps.mean(dim=0)
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
                row["Opt. Gap (%)"] = gap_t[t].item()
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
            global_row["Opt. Gap (%)"] = all_gaps.mean().item()

            # Calculate system-wide totals across all test samples, nodes, and timesteps
            total_pd_pred = all_pred_bus[..., 0].sum().item()
            total_pd_true = all_tgt_bus[..., 0].sum().item()
            total_pg_pred = all_pred_gen[..., 0].sum().item()
            total_pg_true = all_tgt_gen[..., 0].sum().item()

            # Save the specific global metrics requested for the refactored metrics.csv
            custom_csv_data[dataset_name] = {
                "mae_pd": var_metrics[0]["mae"],
                "rmse_pd": var_metrics[0]["rmse"],
                "mae_pg_gen": var_metrics[5]["mae"],
                "rmse_pg_gen": var_metrics[5]["rmse"],
                "total_pd_pred": total_pd_pred,
                "total_pd_true": total_pd_true,
                "total_pg_pred": total_pg_pred,
                "total_pg_true": total_pg_true,
            }
            index_labels = [f"t+{t+1}" for t in range(n_horizon)] + ["GLOBAL"]
            df = pd.DataFrame(rows_per_t + [global_row], index=index_labels)
            df.index.name = "Horizon"

            forecast_csv_path = os.path.join(test_dir, f"{dataset_name}_forecast.csv")
            df.to_csv(forecast_csv_path)

            # ── Plot Forecast vs True for Bus 4 (index 3) ──
            import matplotlib.pyplot as plt
            bus_idx = 3  # "Bus 4" is a load bus in case14
            if all_pred_bus.size(1) > bus_idx:
                horizon_step = 0  # Plot the first step of the forecast (1-step-ahead) across all test samples
                pred_pd = all_pred_bus[:, bus_idx, horizon_step, 0].cpu().numpy()
                tgt_pd = all_tgt_bus[:, bus_idx, horizon_step, 0].cpu().numpy()
                
                plt.figure(figsize=(10, 6))
                plt.plot(tgt_pd, label="True Pd", alpha=0.8)
                plt.plot(pred_pd, label="Forecast Pd", alpha=0.6)
                plt.xlabel("Test Set Timesteps")
                plt.ylabel("Active Load Pd (MW)")
                plt.title(f"Forecast vs True Pd - {dataset_name} Bus 4 (1-step ahead over full test set)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_path = os.path.join(test_dir, f"{dataset_name}_bus4_pd_plot.png")
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()

            # ── Diagnostic plots (time-series + parity) for selected buses ──
            from gridfm_graphkit.tasks.st_forecast_plots import generate_diagnostic_plots

            forecast_horizon = getattr(self.args.data, "forecast_horizon", 1)

            if self._bus_type_pq is not None:
                generate_diagnostic_plots(
                    all_pred_bus, all_tgt_bus,
                    all_pred_gen, all_tgt_gen,
                    self._bus_type_pq, self._bus_type_pv,
                    self._gen_to_bus, self._N_bus,
                    forecast_horizon, dataset_name, test_dir,
                )

        # Cleanup
        self.forecast_preds = {i: [] for i in range(len(self.args.data.networks))}
        self.forecast_targets = {i: [] for i in range(len(self.args.data.networks))}
        self.forecast_naive = {i: [] for i in range(len(self.args.data.networks))}
        self.forecast_gaps = {i: [] for i in range(len(self.args.data.networks))}
        self._bus_type_pq = None
        self._bus_type_pv = None
        self._gen_to_bus = None
        self._N_bus = None

        # Delegate RMSE.csv + metrics.csv (OPF physics metrics) to parent
        super().on_test_end()

        # Extract timing statistics
        train_time_total = getattr(self, "train_time_total_s", 0.0)
        train_samples_count = getattr(self, "train_samples", 0)
        if train_time_total > 0.0 and train_samples_count > 0:
            t_time_val = train_time_total
            t_time_per_sample = train_time_total / train_samples_count
        else:
            t_time_val = " "
            t_time_per_sample = " "

        infer_time_total = getattr(self, "_infer_time_s", 0.0)
        test_sequences_count = getattr(self, "total_test_sequences", 0)
        if test_sequences_count > 0:
            i_time_val = infer_time_total
            i_time_per_sample = infer_time_total / test_sequences_count
        else:
            i_time_val = " "
            i_time_per_sample = " "

        # ── Overwrite metrics.csv with the canonical layout (compare.py-aligned
        #    names + constraint-violation quintet), so the E2E arm is directly
        #    comparable to the exact and surrogate arms. ──
        # NOTE: this writer is @rank_zero_only, so violation stats reflect rank 0's
        # test shard — consistent with the forecast metrics above (single-device
        # test is assumed, as elsewhere in this task).
        final_metrics = self.trainer.callback_metrics
        for dataset_name, custom_data in custom_csv_data.items():
            idx = self.args.data.networks.index(dataset_name)

            # Group this dataset's base OPF metrics for canonical_scalar_rows.
            grouped = {}
            prefix = f"{dataset_name}/"
            for full_key, value in final_metrics.items():
                if full_key.startswith(prefix):
                    grouped[full_key[len(prefix):]] = (
                        value.item() if hasattr(value, "item") else value
                    )

            rows = canonical_scalar_rows(grouped)
            # Forecast-specific rows (not in the OPF callback metrics)
            rows += [
                {"Metric": "MAE Pd", "Value": custom_data["mae_pd"], "Unit": "MW"},
                {"Metric": "RMSE Pd", "Value": custom_data["rmse_pd"], "Unit": "MW"},
                {"Metric": "Generator Pg MAE", "Value": custom_data["mae_pg_gen"], "Unit": "MW"},
            ]
            # Canonical constraint-violation quintet
            rows += self._viol_acc[idx].finalize_rows(prefix="")
            # E2E diagnostics + timing (kept)
            rows += [
                {"Metric": "total_pd_true", "Value": custom_data["total_pd_true"], "Unit": "MW"},
                {"Metric": "total_pd_pred", "Value": custom_data["total_pd_pred"], "Unit": "MW"},
                {"Metric": "total_pg_true", "Value": custom_data["total_pg_true"], "Unit": "MW"},
                {"Metric": "total_pg_pred", "Value": custom_data["total_pg_pred"], "Unit": "MW"},
                {"Metric": "Training time model-only (s)", "Value": t_time_val, "Unit": "s"},
                {"Metric": "Training time model-only per sample (s)", "Value": t_time_per_sample, "Unit": "s"},
                {"Metric": "Inference time model-only (s)", "Value": i_time_val, "Unit": "s"},
                {"Metric": "Inference time model-only per sample (s)", "Value": i_time_per_sample, "Unit": "s"},
            ]

            residuals_csv_path = os.path.join(test_dir, f"{dataset_name}_metrics.csv")
            pd.DataFrame(rows).to_csv(residuals_csv_path, index=False)

        # Reset violation accumulators for any subsequent test run.
        self._viol_acc = {i: ConstraintViolationAccumulator() for i in range(len(self.args.data.networks))}




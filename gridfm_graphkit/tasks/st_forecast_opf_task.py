"""
ST-GNN Forecast OPF Task — PyTorch Lightning module for the
Space-then-Time multi-period AC-OPF forecaster.

Inherits from ReconstructionTask but overrides:
    - __init__: instantiates ST_GNN_heterogeneous and ST losses
    - forward: unpacks collate_temporal dict
    - _shared_step / training_step / validation_step: dual-batch handling
    - test_step / predict_step: placeholders for Phase 6

The dataloader yields dicts from collate_temporal:
    {"folded_batch", "target_batch", "B", "W", "n", "N_bus"}
"""

import torch
from pytorch_lightning.utilities import rank_zero_only

from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.base_task import BaseTask
from gridfm_graphkit.models.st_gnn_heterogeneous import ST_GNN_heterogeneous
from gridfm_graphkit.training.loss import ST_ForecastBusMSE, ST_ForecastGenMSE, ST_PhysicsForecastLoss
from gridfm_graphkit.datasets.globals import PD_H, QD_H, QG_H, VM_H, VA_H, PG_H


# Target feature indices for constructing the [B, N, n, 5] target tensor
BUS_TARGET_INDICES = [PD_H, QD_H, QG_H, VM_H, VA_H]  # -> [Pd, Qd, Qg, Vm, Va]
GEN_TARGET_INDICES = [PG_H]                             # -> [Pg]


@TASK_REGISTRY.register("ST_ForecastOPF")
class ST_ForecastOPFTask(BaseTask):
    """
    PyTorch Lightning task for ST-GNN multi-period AC-OPF forecasting.

    Handles the dual-batch dataloader output from collate_temporal and
    composes weighted MSE (load + OPF) and physics forecast losses.
    """

    def __init__(self, args, data_normalizers):
        super().__init__(args, data_normalizers)

        # ---- Model ----
        self.model = ST_GNN_heterogeneous(
            args,
            use_exogenous=getattr(args.model, "use_exogenous", False),
        )

        # ---- Loss functions ----
        # Create a simple namespace for loss_args (loss-specific config)
        loss_args = _get_loss_args(args)

        self.mse_bus_loss_fn = ST_ForecastBusMSE(loss_args, args)
        self.mse_gen_loss_fn = ST_ForecastGenMSE(loss_args, args)
        self.physics_loss_fn = ST_PhysicsForecastLoss(loss_args, args)

        # Loss weights
        self.lambda_mse = getattr(loss_args, "lambda_mse", 1.0)
        self.lambda_gen = getattr(loss_args, "lambda_gen", 0.1)
        self.lambda_phys = getattr(loss_args, "lambda_phys", 0.1)

        self.batch_size = int(args.training.batch_size)
        self.test_outputs = {i: [] for i in range(len(args.data.networks))}

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

        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        total_loss, log_dict = self._shared_step(batch, prefix="train")

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
        total_loss, log_dict = self._shared_step(batch, prefix="val")

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
    # Test / Predict (Phase 6 placeholders)
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        total_loss, log_dict = self._shared_step(batch, prefix="test")
        self.log_dict(
            log_dict,
            batch_size=batch["B"],
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    @rank_zero_only
    def on_test_end(self):
        self.test_outputs.clear()


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

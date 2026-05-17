import torch.nn.functional as F
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gridfm_graphkit.io.registries import LOSS_REGISTRY
from torch_scatter import scatter_add

from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    QG_H,
    VM_H,
    VA_H,
    QD_H,
    PD_H,
    # Output feature indices
    VM_OUT,
    VA_OUT,
    QG_OUT,
    PG_OUT,
    # Generator feature indices
    PG_H,
)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all custom loss functions.
    """

    @abstractmethod
    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        """
        Compute the loss.

        Parameters:
        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.
        - model: Optional model reference for accessing internal states.

        Returns:
        - A dictionary with the total loss and any additional metrics.
        """
        pass


@LOSS_REGISTRY.register("MaskedMSE")
class MaskedMSELoss(BaseLoss):
    """
    Mean Squared Error loss computed only on masked elements.
    """

    def __init__(self, loss_args, args):
        super(MaskedMSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        loss = F.mse_loss(pred[mask], target[mask], reduction=self.reduction)
        return {"loss": loss, "Masked MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedGenMSE")
class MaskedGenMSE(torch.nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
    ):
        loss = F.mse_loss(
            pred_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            target_dict["gen"][mask_dict["gen"][:, : (PG_H + 1)]],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked generator MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MaskedBusMSE")
class MaskedBusMSE(torch.nn.Module):
    def __init__(self, loss_args, args):
        super().__init__()
        self.reduction = "mean"
        self.args = args

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
    ):
        if self.args.task == "OptimalPowerFlow":
            pred_cols = [VM_OUT, VA_OUT, QG_OUT]
            target_cols = [VM_H, VA_H, QG_H]
        else:
            pred_cols = [VM_OUT, VA_OUT]
            target_cols = [VM_H, VA_H]

        pred_bus = pred_dict["bus"][:, pred_cols]  # shape: [N, 3]
        target_bus = target_dict["bus"][:, target_cols]

        mask = mask_dict["bus"][:, target_cols]

        loss = F.mse_loss(
            pred_bus[mask],
            target_bus[mask],
            reduction=self.reduction,
        )
        return {"loss": loss, "Masked bus MSE loss": loss.detach()}


@LOSS_REGISTRY.register("MSE")
class MSELoss(BaseLoss):
    """Standard Mean Squared Error loss."""

    def __init__(self, loss_args, args):
        super(MSELoss, self).__init__()
        self.reduction = "mean"

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return {"loss": loss, "MSE loss": loss.detach()}


class MixedLoss(BaseLoss):
    """
    Combines multiple loss functions with weighted sum.

    Args:
        loss_functions (list[nn.Module]): List of loss functions.
        weights (list[float]): Corresponding weights for each loss function.
    """

    def __init__(self, loss_functions, weights):
        super(MixedLoss, self).__init__()

        if len(loss_functions) != len(weights):
            raise ValueError(
                "The number of loss functions must match the number of weights.",
            )

        self.loss_functions = nn.ModuleList(loss_functions)
        self.weights = weights

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        """
        Compute the weighted sum of all specified losses.

        Parameters:

        - pred: Predictions.
        - target: Ground truth.
        - edge_index: Optional edge index for graph-based losses.
        - edge_attr: Optional edge attributes for graph-based losses.
        - mask: Optional mask to filter the inputs for certain losses.

        Returns:
        - A dictionary with the total loss and individual losses.
        """
        total_loss = 0.0
        loss_details = {}

        for i, loss_fn in enumerate(self.loss_functions):
            loss_output = loss_fn(
                pred,
                target,
                edge_index,
                edge_attr,
                mask,
                model,
            )

            # Assume each loss function returns a dictionary with a "loss" key
            individual_loss = loss_output.pop("loss")
            weighted_loss = self.weights[i] * individual_loss

            total_loss += weighted_loss

            # Add other keys from the loss output to the details
            for key, val in loss_output.items():
                loss_details[key] = val

        loss_details["loss"] = total_loss
        return loss_details


def _resolve_horizon_weights(weighting, horizon_len, device, dtype):
    weighting = str(weighting).lower()
    if weighting == "uniform":
        return None
    if weighting == "harmonic":
        tau = 0.8
        steps = torch.arange(1, horizon_len + 1, device=device, dtype=dtype)
        return 1.0 / (steps ** tau)
    raise ValueError(f"Unknown horizon_weighting: {weighting}")




@LOSS_REGISTRY.register("LayeredWeightedPhysics")
class LayeredWeightedPhysicsLoss(BaseLoss):
    def __init__(self, loss_args, args) -> None:
        super().__init__()
        self.base_weight = loss_args.base_weight

    def forward(
        self,
        pred,
        target,
        edge_index=None,
        edge_attr=None,
        mask=None,
        model=None,
    ):
        total_loss = 0.0
        loss_details = {}

        layer_keys = sorted(model.layer_residuals.keys())
        L = len(layer_keys)

        # Compute raw weights (geometric decay)
        raw_weights = [self.base_weight ** (L - idx - 1) for idx in range(L)]

        # Normalize so weights sum to 1
        weight_sum = sum(raw_weights)
        norm_weights = [w / weight_sum for w in raw_weights]

        for key, weight in zip(layer_keys, norm_weights):
            residual = model.layer_residuals[key]
            total_loss = total_loss + weight * residual
            loss_details[f"layer_{key}_residual"] = residual.item()
            loss_details[f"layer_{key}_weight"] = weight

        loss_details["loss"] = total_loss
        loss_details["Layered Weighted Physics Loss"] = total_loss.item()
        return loss_details


@LOSS_REGISTRY.register("LossPerDim")
class LossPerDim(BaseLoss):
    def __init__(self, loss_args, args):
        super(LossPerDim, self).__init__()
        self.reduction = "mean"
        self.loss_str = loss_args.loss_str
        self.dim = loss_args.dim
        if self.dim not in ["VM", "VA", "P_in", "Q_in"]:
            raise ValueError(
                f"LossPerDim initialized with not valid dim: {self.dim}",
            )

        elif self.loss_str not in ["MAE", "MSE"]:
            raise ValueError(
                f"LossPerDim initialized with not valid loss_str: {self.loss_str}",
            )

    def forward(
        self,
        pred_dict,
        target_dict,
        edge_index,
        edge_attr,
        mask_dict,
        model=None,
    ):
        if self.dim == "VM":
            temp_pred = pred_dict["bus"][:, VM_OUT]
            temp_target = target_dict["bus"][:, VM_H]
        elif self.dim == "VA":
            temp_pred = pred_dict["bus"][:, VA_OUT]
            temp_target = target_dict["bus"][:, VA_H]
        elif self.dim == "P_in":
            temp_pred = pred_dict["bus"][:, PG_OUT]
            num_bus = temp_pred.size(0)
            gen_to_bus_index = edge_index[("gen", "connected_to", "bus")]
            temp_gen = scatter_add(
                target_dict["gen"][:, PG_H],
                gen_to_bus_index[1, :],
                dim=0,
                dim_size=num_bus,
            )
            temp_target = temp_gen - target_dict["bus"][:, PD_H]
        elif self.dim == "Q_in":
            temp_pred = pred_dict["bus"][:, QG_OUT]
            temp_target = target_dict["bus"][:, QG_H] - target_dict["bus"][:, QD_H]

        mse_loss = F.mse_loss(temp_pred, temp_target, reduction=self.reduction)
        mae_loss = F.l1_loss(temp_pred, temp_target, reduction=self.reduction)

        loss = mse_loss if self.loss_str == "mse" else mae_loss
        return {
            "loss": loss,
            f"MSE loss {self.dim}": mse_loss.detach(),
            f"MAE loss {self.dim}": mae_loss.detach(),
        }

@LOSS_REGISTRY.register("ForecastBusMSE")
class ForecastBusMSE(BaseLoss):
    def __init__(self, loss_args, args):
        super().__init__()

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None):
        loss_bus = F.mse_loss(pred["bus"], target["bus"], reduction="mean")
        return {"loss": loss_bus, "Forecast bus MSE": loss_bus.detach()}


@LOSS_REGISTRY.register("ForecastGenMSE")
class ForecastGenMSE(BaseLoss):
    def __init__(self, loss_args, args):
        super().__init__()

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None):
        loss_gen = F.mse_loss(pred["gen"], target["gen"], reduction="mean")
        return {"loss": loss_gen, "Forecast gen MSE": loss_gen.detach()}


# ======================================================================
# ST-GNN specific losses
# ======================================================================

@LOSS_REGISTRY.register("ST_ForecastBusMSE")
class ST_ForecastBusMSE(BaseLoss):
    """
    Weighted MSE for ST-GNN bus forecast output.

    Splits the 5-dim bus prediction [Pd, Qd, Qg, Vm, Va] into:
        - load component (Pd, Qd):  indices [0:2], weighted by lambda_load
        - OPF component  (Qg, Vm, Va): indices [2:5], weighted by lambda_opf

    Expected shapes:
        pred["bus"]:   [B, N_bus, n, 5]
        target["bus"]: [B, N_bus, n, 5]  (same layout: [Pd, Qd, Qg, Vm, Va])
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.lambda_load = getattr(loss_args, "lambda_load", 0.5)
        self.lambda_opf = getattr(loss_args, "lambda_opf", 0.5)
        self.horizon_weighting = getattr(args.training, "horizon_weighting", "uniform")

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None):
        pred_load = pred["bus"][..., 0:2]    # [B, N_bus, n, 2]: Pd, Qd
        pred_opf = pred["bus"][..., 2:5]     # [B, N_bus, n, 3]: Qg, Vm, Va

        target_load = target["bus"][..., 0:2]
        target_opf = target["bus"][..., 2:5]

        # Descale into stable Global P.U. space when window normalizer is used
        if mask is not None and mask.get("window_baseMVA") is not None:
            baseMVA = mask["window_baseMVA"].view(-1, 1, 1, 1).to(pred["bus"].device) 
            scaled_baseMVA = baseMVA / mask["static_baseMVA"]
            pred_load = pred_load * scaled_baseMVA
            target_load = target_load * scaled_baseMVA

        def weighted_mse(pred_t, target_t):
            err = pred_t - target_t
            mse_per_t = (err ** 2).mean(dim=(0, 1, 3))
            weights = _resolve_horizon_weights(
                self.horizon_weighting, mse_per_t.numel(), pred_t.device, pred_t.dtype,
            )
            if weights is None:
                return mse_per_t.mean()
            weight_sum = weights.sum()
            if weight_sum <= 0:
                raise ValueError("horizon weights must sum to a positive value.")
            weights = weights / weight_sum
            return (mse_per_t * weights).sum()

        loss_load = weighted_mse(pred_load, target_load)
        loss_opf = weighted_mse(pred_opf, target_opf)

        total = self.lambda_load * loss_load + self.lambda_opf * loss_opf

        return {
            "loss": total,
            "ST load MSE (Pd,Qd)": loss_load.detach(),
            "ST OPF MSE (Qg,Vm,Va)": loss_opf.detach(),
        }


@LOSS_REGISTRY.register("ST_ForecastGenMSE")
class ST_ForecastGenMSE(BaseLoss):
    """
    MSE for ST-GNN gen forecast output.

    Expected shapes:
        pred["gen"]:   [B, N_gen, n, 1]
        target["gen"]: [B, N_gen, n, 1]
    """

    def __init__(self, loss_args, args):
        super().__init__()
        self.horizon_weighting = getattr(args.training, "horizon_weighting", "uniform")

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None):
        # Descale into stable Global P.U. space when window normalizer is used
        pred_gen = pred["gen"]
        target_gen = target["gen"]
        if mask is not None and mask.get("window_baseMVA") is not None:
            baseMVA = mask["window_baseMVA"].view(-1, 1, 1, 1).to(pred_gen.device)
            scaled_baseMVA = baseMVA / mask["static_baseMVA"]
            pred_gen = pred_gen * scaled_baseMVA
            target_gen = target_gen * scaled_baseMVA

        err = pred_gen - target_gen
        mse_per_t = (err ** 2).mean(dim=(0, 1, 3))
        weights = _resolve_horizon_weights(
            self.horizon_weighting, mse_per_t.numel(), pred_gen.device, pred_gen.dtype,
        )
        if weights is None:
            loss_gen = mse_per_t.mean()
        else:
            weight_sum = weights.sum()
            if weight_sum <= 0:
                raise ValueError("horizon weights must sum to a positive value.")
            weights = weights / weight_sum
            loss_gen = (mse_per_t * weights).sum()

        return {"loss": loss_gen, "ST gen MSE (Pg)": loss_gen.detach()}


@LOSS_REGISTRY.register("ST_OptimalityGapLoss")
class ST_OptimalityGapLoss(BaseLoss):
    """
    Minimizes the optimality gap defined by the difference between predicted and true generation costs.
    cost is C(Pg) = c0 + c1 * Pg + c2 * Pg ** 2
    The loss computes the MSE (or MAE) of (C(Pg_pred) - C(Pg_true)).
    """

    def __init__(self, loss_args, args):
        super().__init__()

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None):
        from gridfm_graphkit.datasets.globals import C0_H, C1_H, C2_H

        c0_norm = mask["gen_x_4d"][..., C0_H:C0_H + 1]  # [B, N_gen, n, 1]
        c1_norm = mask["gen_x_4d"][..., C1_H:C1_H + 1]
        c2_norm = mask["gen_x_4d"][..., C2_H:C2_H + 1]

        c0_unnorm = torch.sign(c0_norm) * (torch.exp(torch.abs(c0_norm)) - 1.0)
        c1_unnorm = torch.sign(c1_norm) * (torch.exp(torch.abs(c1_norm)) - 1.0)
        c2_unnorm = torch.sign(c2_norm) * (torch.exp(torch.abs(c2_norm)) - 1.0)        

        # Convert predictions/targets from p.u. back to MW for the cost function
        baseMVA = mask.get("window_baseMVA", None)
        if baseMVA is not None:
            # Reshape [B] -> [B, 1, 1, 1] to broadcast against [B, N_gen, n, 1]
            baseMVA = baseMVA.view(mask["B"], 1, 1, 1).to(pred["gen"].device)
        else:
            baseMVA = 1 # dont denormalize ...

        pred_pg_mw = pred["gen"] * baseMVA
        target_pg_mw = target["gen"] * baseMVA

        pred_cost = c0_unnorm + c1_unnorm * pred_pg_mw + c2_unnorm * (pred_pg_mw ** 2)
        target_cost = c0_unnorm + c1_unnorm * target_pg_mw + c2_unnorm * (target_pg_mw ** 2)
        
        cost_loss = F.mse_loss(pred_cost, target_cost, reduction="mean")
        
        return {
            "loss": cost_loss,
            "Optimality Gap (MSE)": cost_loss.detach()
        }


@LOSS_REGISTRY.register("ST_PhysicsForecastLoss")
class ST_PhysicsForecastLoss(BaseLoss):
    """
    Physics-informed loss for ST-GNN forecast output.

    - Flattens the multi-step temporal predictions to match static graph topologies.
    - Injects the predicted load (Pd, Qd) into the physical bus features to evaluate a perfectly self-consistent physical state.
    - Runs predictions through the physics layers (ComputeBranchFlow, 
      ComputeNodeInjection, ComputeNodeResiduals).
    -> Computes the active and reactive power balance residuals (delta_P, delta_Q).

    Args:
        pred (dict):
            - "bus": [B, N_bus, n, 5] forecast [Pd, Qd, Qg, Vm, Va]
            - "gen": [B, N_gen, n, 1] forecast [Pg]
        edge_index (dict): from target_batch.edge_index_dict
        edge_attr (dict):  from target_batch.edge_attr_dict
        mask (dict): containing:
            - "bus_x": [B*n*N_bus, F_bus] original target bus features (provides static GS, BS)
            - "B":     int, batch size
            - "n":     int, forecast horizon
            - "N_bus": int, buses per graph

    Returns: 
        dict: mean squared power balance residuals.
    """

    def __init__(self, loss_args, args):
        super().__init__()
        from gridfm_graphkit.models.utils import (
            ComputeBranchFlow,
            ComputeNodeInjection,
            ComputeNodeResiduals,
        )
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()

    def forward(self, pred, target, edge_index=None, edge_attr=None, mask=None, model=None):
        B = mask["B"]
        n = mask["n"]
        N_bus = mask["N_bus"]
        bus_x_orig = mask["bus_x"]  # [B*n*N_bus, F_bus]

        # --- Flatten predictions to [B*n*N, ...] matching target_batch topology ---
        pred_bus_flat = pred["bus"].permute(0, 2, 1, 3).reshape(-1, 5) # [B, N_bus, n, 5]  -> [B*n*N_bus, 5]
        pred_gen_flat = pred["gen"].permute(0, 2, 1, 3).reshape(-1, 1) # [B, N_gen, n, 1] -> [B*n*N_gen, 1]

        # Var. Extraction
        Pd_pred = pred_bus_flat[:, 0] 
        Qd_pred = pred_bus_flat[:, 1]   
        Vm_pred = pred_bus_flat[:, 3]   
        Va_pred = pred_bus_flat[:, 4]   
        Qg_pred = pred_bus_flat[:, 2]

        # add Pg from generators to buses
        edge_index_gb = edge_index[("gen", "connected_to", "bus")]
        num_bus_total = B * n * N_bus
        Pg_bus_pred = scatter_add(
            pred_gen_flat[:, 0],
            edge_index_gb[1],
            dim=0,
            dim_size=num_bus_total,
        )

        # Create Bus pred tensor in correct format:  [B*n*N_bus, 4] matching [VM_OUT, VA_OUT, PG_OUT, QG_OUT]
        bus_data_pred = torch.stack([Vm_pred, Va_pred, Pg_bus_pred, Qg_pred], dim=-1) 
        
        # ---------------------------------------------------------
        # Create  Self-Consistent Predicted Universe
        # ---------------------------------------------------------
        # Clone to avoid modifying the original target tensor in-place
        bus_x_physics = bus_x_orig.clone() 
        
        # Overwrite  ground truth load with PREDICTED load
        bus_x_physics[:, PD_H] = Pd_pred
        bus_x_physics[:, QD_H] = Qd_pred

        # --- Physics pipeline ---
        edge_index_bb = edge_index[("bus", "connects", "bus")]
        edge_attr_bb = edge_attr[("bus", "connects", "bus")]

        Pft, Qft = self.branch_flow_layer(bus_data_pred, edge_index_bb, edge_attr_bb)
        P_in, Q_in = self.node_injection_layer(Pft, Qft, edge_index_bb, num_bus_total)
        res_P, res_Q = self.node_residuals_layer(P_in, Q_in, bus_data_pred, bus_x_physics)

        # For sample-wise normalizer: Scale the resulting p.u. residuals into stable Global P.U. space
        if mask is not None and mask.get("window_baseMVA") is not None:
            scale = mask["window_baseMVA"] / mask["static_baseMVA"]
            b_bus_scale = scale.repeat_interleave(n * N_bus).squeeze().to(res_P.device)
            res_P = res_P * b_bus_scale
            res_Q = res_Q * b_bus_scale

        loss = torch.mean(res_P ** 2 + res_Q ** 2)

        return {
            "loss": loss,
            "ST Physics Loss": loss.detach(),
        }


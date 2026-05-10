"""
Standalone Spatio-Temporal GNN for MP-ACOPF forecasting.

Contains two classes:
    1. UnmaskedSpatialEncoder - lightweight spatial backbone (no physics/MLPs)
    2. ST_GNN_heterogeneous   - end-to-end forecaster composing spatial encoder,
                                temporal TCN, optional late fusion, and forecast decoders

The spatial encoder mirrors the TransformerConv layers of GNS_heterogeneous
but strips away masking, physics decoders, and per-layer MLP decoding.
This avoids VRAM bloat on folded (B*W) batches while preserving the
message-passing architecture for potential weight transfer.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv

from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.utils import bound_with_sigmoid
from gridfm_graphkit.models.temporal_encoders import TCN
from gridfm_graphkit.datasets.globals import (
    PD_H, QD_H,
    MIN_VM_H, MAX_VM_H,
    MIN_PG, MAX_PG,
)



# Forecast output indices (5-dim bus: [Pd, Qd, Qg, Vm, Va])
FORECAST_VM_IDX = 3
FORECAST_PG_IDX = 0

# Default exogenous feature indices from bus x_dict
DEFAULT_EXO_BUS_INDICES = [PD_H, QD_H]


# ======================================================================
# 1. Unmasked Spatial Encoder
# ======================================================================

class UnmaskedSpatialEncoder(nn.Module):
    """
    Lightweight spatial backbone mirroring GNS_heterogeneous but stripped
    of all physics decoders, per-layer MLPs, and masking logic.

    Accepts fully observable x_dict (all features known for historical
    timesteps) and produces latent embeddings h_bus and h_gen.

    Architecture matches GNS_heterogeneous conv layers exactly:
        - input_proj_{bus,gen,edge}: 2-layer MLP + LayerNorm
        - HeteroConv(TransformerConv) per relation, with skip connections
        - LayerNorm per node type per layer

    This allows weight transfer from a pretrained GNS_heterogeneous checkpoint.
    """

    def __init__(self, args):
        super().__init__()

        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.edge_dim = args.model.edge_dim
        self.heads = args.model.attention_head
        self.dropout = getattr(args.model, "dropout", 0.0)

        self.latent_dim = self.hidden_dim * self.heads  # final embedding dim

        # --- Input projections (identical to GNS_heterogeneous) ---
        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # --- HeteroConv layers (identical to GNS_heterogeneous) ---
        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()

        self.activation = nn.LeakyReLU()

        for i in range(self.num_layers):
            # First layer: hidden_dim input; subsequent: hidden_dim * heads
            in_bus = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            in_gen = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            out_dim = self.hidden_dim

            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_bus, out_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen, out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus, out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }

            self.layers.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms_bus.append(nn.LayerNorm(out_dim * self.heads))
            self.norms_gen.append(nn.LayerNorm(out_dim * self.heads))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Spatial message-passing on fully observable heterogeneous graphs.
        No masking, no physics, no MLP decoding.

        x_dict: {"bus": [N_bus, F_bus], "gen": [N_gen, F_gen]}
        edge_index_dict: relation-keyed edge indices
        edge_attr_dict: relation-keyed edge attributes (bus-bus requires G,B)

        Returns:
            h_bus: [N_bus, latent_dim]  (latent_dim = hidden_dim * heads)
            h_gen: [N_gen, latent_dim]
        """
        # Project inputs
        h_bus = self.input_proj_bus(x_dict["bus"])
        h_gen = self.input_proj_gen(x_dict["gen"])

        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        # Message-passing layers with skip connections
        for i, conv in enumerate(self.layers):
            out_dict = conv(
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj_dict,
            )
            out_bus = out_dict["bus"]
            out_gen = out_dict["gen"]

            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # Skip connection (first layer may change dim: hidden_dim -> hidden_dim * heads)
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

        return h_bus, h_gen


# ======================================================================
# 2. ST-GNN End-to-End Forecaster
# ======================================================================

@MODELS_REGISTRY.register("ST_GNN_heterogeneous")
class ST_GNN_heterogeneous(nn.Module):
    """
    Space-then-Time heterogeneous GNN for MP-ACOPF forecasting.

    End-to-end model: predicts exogenous loads AND optimal dispatch
    at the target horizon.

    Bus output (5 features): [Pd, Qd, Qg, Vm, Va]
    Gen output (1 feature):  [Pg]

    Pipeline:
        folded_batch [B*W graphs]
          -> UnmaskedSpatialEncoder   -> h_bus, h_gen
          -> unfold                   -> [B, N, W, D]
          -> TCN                      -> z_bus, z_gen  [B, N, D]
          -> expand to n steps        -> [B, N, n, D]
          -> (optional) late fusion   -> [z || C_exo]
          -> forecast MLP             -> y_bus [B, N_bus, n, 5], y_gen [B, N_gen, n, 1]
          -> bound Vm, Pg per step    -> final forecast
    """

    def __init__(
        self,
        args,
        exo_bus_indices: list = None,
        exo_gen_dim: int = 0,
    ):
        super().__init__()

        self.use_exogenous = getattr(args.model, "use_exogenous", False)
        self.exo_bus_indices = exo_bus_indices or DEFAULT_EXO_BUS_INDICES
        self.exo_gen_dim = exo_gen_dim
        self.n = args.data.forecast_horizon  # number of output steps

        # ---- Spatial encoder ----
        self.spatial_encoder = UnmaskedSpatialEncoder(args)

        # ---- Dimensions ----
        hidden_dim = args.model.hidden_size
        heads = args.model.attention_head
        self.latent_dim = hidden_dim * heads

        self.forecast_bus_dim = args.model.output_bus_dim  # [Pd, Qd, Qg, Vm, Va]
        self.forecast_gen_dim = args.model.output_gen_dim  # [Pg]

        # ---- Temporal encoders ----
        temporal_window = args.data.temporal_window
        tcn_kernel = getattr(args.model, "tcn_kernel_size", 3)
        tcn_dropout = getattr(args.model, "dropout", 0.0)

        self.tcn_bus = TCN(
            input_dim=self.latent_dim,
            window_size=temporal_window,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
        )

        self.tcn_gen = TCN(
            input_dim=self.latent_dim,
            window_size=temporal_window,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
        )


        # ---- Forecast decoders ----
        # Direct multi-step decoding: maps D -> n * F
        # Bus decoder receives z_bus [B, N_bus, D] concatenated with
        # standardized Pd_macro [B, N_bus, 1] -> input dim = D + PD_MACRO_DIM
        exo_bus_dim = len(self.exo_bus_indices) if self.use_exogenous else 0
        exo_gen_d = self.exo_gen_dim if self.use_exogenous else 0

        mlp_hidden_dim = getattr(args.model, "mlp_hidden_dim", 1024)
        mlp_num_layers = getattr(args.model, "mlp_num_layers", 1)

        def build_decoder(in_dim, out_dim):
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
            ]
            curr_dim = hidden_dim
            for _ in range(mlp_num_layers):
                layers.extend([
                    nn.Linear(curr_dim, mlp_hidden_dim),
                    nn.LayerNorm(mlp_hidden_dim),
                    nn.LeakyReLU(),
                ])
                curr_dim = mlp_hidden_dim
            layers.append(nn.Linear(curr_dim, out_dim))
            return nn.Sequential(*layers)

        # Bus decoder: latent_dim (+ exo if used)
        self.forecast_decoder_bus = build_decoder(
            self.latent_dim + exo_bus_dim, self.forecast_bus_dim * self.n
        )
        # Gen decoder: latent_dim (+ exo if used)
        self.forecast_decoder_gen = build_decoder(
            self.latent_dim + exo_gen_d, self.forecast_gen_dim * self.n
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, folded_batch, target_batch, B, W, N_bus):
        """
        Space-then-Time forward pass.

        folded_batch: PyG Batch of B*W window graphs
        target_batch: PyG Batch of B*n target graphs (exo features + bounds)
        B: batch size
        W: temporal window size
        N_bus: number of bus nodes per graph

        Returns:
            dict with:
                "bus": [B, N_bus, n, 5]  forecast [Pd, Qd, Qg, Vm, Va]
                "gen": [B, N_gen, n, 1]  forecast [Pg]
        """
        D = self.latent_dim
        n = self.n

        # ==============================================================
        # 1. Spatial pass — unmasked message-passing on B*W graphs
        # ==============================================================
        h_bus, h_gen = self.spatial_encoder(
            folded_batch.x_dict,
            folded_batch.edge_index_dict,
            folded_batch.edge_attr_dict,
        )
        # h_bus: [B*W*N_bus, D]    h_gen: [B*W*N_gen, D]

        N_gen = h_gen.size(0) // (B * W)

        # ==============================================================
        # 2. Unfold to [B, N, W, D]
        # ==============================================================
        # collate_temporal uses sample-major, time-minor ordering
        # -> view(B, W, N, D) is valid, then permute to [B, N, W, D]
        h_bus_4d = h_bus.view(B, W, N_bus, D).permute(0, 2, 1, 3)  # [B, N_bus, W, D]
        h_gen_4d = h_gen.view(B, W, N_gen, D).permute(0, 2, 1, 3)  # [B, N_gen, W, D]

        # ==============================================================
        # 3. Temporal pass — TCN -> terminal state
        # ==============================================================
        z_bus = self.tcn_bus(h_bus_4d)  # [B, N_bus, D]
        z_gen = self.tcn_gen(h_gen_4d)  # [B, N_gen, D]

        if self.use_exogenous:
            pass # (Exogenous features skipped for direct decoding for now)

        # ==============================================================
        # 4. Forecast decoders — Direct multi-step decoding
        # ==============================================================
        forecast_bus_flat = self.forecast_decoder_bus(z_bus)  # [B, N_bus, n * 5]
        forecast_gen_flat = self.forecast_decoder_gen(z_gen)  # [B, N_gen, n * 1]
        
        # Reshape the flat sequences into proper time dimension: [B, N, n, F]
        forecast_bus = forecast_bus_flat.view(B, N_bus, n, self.forecast_bus_dim)
        forecast_gen = forecast_gen_flat.view(B, N_gen, n, self.forecast_gen_dim)

        # ==============================================================
        # 6. Bounds per step — Vm (sigmoid) and Pg (sigmoid)
        # ==============================================================
        # target shapes: [B*n*N_bus, F] -> [B, n, N_bus, F] -> [B, N_bus, n, F]
        target_bus_x_full = target_batch["bus"].x.view(B, n, N_bus, -1).permute(0, 2, 1, 3)
        target_gen_x_full = target_batch["gen"].x.view(B, n, N_gen, -1).permute(0, 2, 1, 3)

        min_vm = target_bus_x_full[..., MIN_VM_H]  # [B, N_bus, n]
        max_vm = target_bus_x_full[..., MAX_VM_H]
        min_pg = target_gen_x_full[..., MIN_PG]    # [B, N_gen, n]
        max_pg = target_gen_x_full[..., MAX_PG]

        #  out-of-place ops (torch.cat / reassignment) instead of in-place index
        # assignment because in place leads to -> runtimeError (version-counter conflicts during backpropagation)
        forecast_gen = bound_with_sigmoid(
            forecast_gen[..., FORECAST_PG_IDX], min_pg, max_pg,
        ).unsqueeze(-1)
        vm_bounded = bound_with_sigmoid(
            forecast_bus[..., FORECAST_VM_IDX], min_vm, max_vm,
        ).unsqueeze(-1)
        forecast_bus = torch.cat([
            forecast_bus[..., :FORECAST_VM_IDX],
            vm_bounded,
            forecast_bus[..., FORECAST_VM_IDX + 1:],
        ], dim=-1)

        return {
            "bus": forecast_bus,  # [B, N_bus, n, 5]: [Pd, Qd, Qg, Vm, Va]
            "gen": forecast_gen,  # [B, N_gen, n, 1]: [Pg]
        }

"""
Standalone Spatio-Temporal GNN for MP-ACOPF forecasting.

Contains two classes:
    1. UnmaskedSpatialEncoder - spatial backbone (no physics/MLPs)
    2. ST_GNN_heterogeneous   - end-to-end forecaster composing spatial encoder,
                                temporal TCN, optional late fusion, and forecast decoders

The spatial encoder mirrors the TransformerConv layers of GNS_heterogeneous
but strips away masking, physics decoders, and per-layer MLP decoding.
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


class CrossAttentionTimeDecoder(nn.Module):
    """
    Replaces Linear(W, n) time projection with learnable-query cross-attention.
    Query = learnable horizon positional embeddings (E_future) shifted by a
    dynamic context vector derived from the terminal TCN state h_W.
    Uses Pre-LN formulation with a residual that bypasses both LayerNorm
    and attention, giving E_future a direct gradient flow to the loss.
    Args:
        latent_dim:  D — must be divisible by num_heads
        horizon:     n — number of forecast steps
        num_heads:   temporal_decoder_heads (config)
        dropout:     attention dropout
    """
    def __init__(self, latent_dim: int, horizon: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        # Learnable positional baseline for each future step: [1, n, D]
        self.E_future = nn.Parameter(torch.empty(1, horizon, latent_dim))
        nn.init.trunc_normal_(self.E_future, std=0.02)
        # Projects terminal TCN state -> dynamic context shift [B*N, 1, D]
        self.context_proj = nn.Linear(latent_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Pre-LN: normalise query before attention, not after
        self.norm_q = nn.LayerNorm(latent_dim)
    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: [B, N, D, W]  — TCN output (features-first, time-last)
        returns: [B, N, n, D]
        """
        B, N, D, W = z_seq.shape
        # --- K and V: full TCN sequence [B*N, W, D] ---
        kv = z_seq.permute(0, 1, 3, 2).contiguous().reshape(B * N, W, D)
        # --- Dynamic context from terminal TCN state h_W ---
        # z_seq[:, :, :, -1] -> [B, N, D] -> [B*N, D]
        terminal_state = z_seq[:, :, :, -1].contiguous().reshape(B * N, D)
        # Project and broadcast across all n horizon steps: [B*N, 1, D]
        context = self.context_proj(terminal_state).unsqueeze(1)
        # --- Query: positional baseline + dynamic shift ---
        # q_base: [1, n, D] -> expand [B*N, n, D] (view, no copy)
        # q:      [B*N, n, D]  (materialised by the addition)
        q = self.E_future.expand(B * N, -1, -1) + context
        # --- Pre-LN cross-attention ---
        # Normalise query before attention; residual bypasses both norm and attn
        attn_out, _ = self.cross_attn(self.norm_q(q), kv, kv)  # [B*N, n, D]
        out = q + attn_out                                       # [B*N, n, D]
        return out.reshape(B, N, self.horizon, D)

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
          -> TCN sequence             -> z_bus_seq, z_gen_seq  [B, N, D, W]
          -> Time Projection          -> z_bus_future [B, N, D, n]
          -> transpose                -> [B, N, n, D]
          -> (optional) late fusion   -> [z || C_exo]
          -> forecast MLP (per step)  -> y_bus [B, N, n, 5], y_gen [B, N, n, 1]
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
        self.temporal_decoder = getattr(args.model, "temporal_decoder", "cross_attention")

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

        # ---- Temporal decoder ----
        if self.temporal_decoder == "cross_attention":
            decoder_heads = args.model.temporal_decoder_heads
            assert self.latent_dim % decoder_heads == 0, (
                f"latent_dim ({self.latent_dim}) must be divisible by "
                f"temporal_decoder_heads ({decoder_heads})"
            )
            self.time_attn_bus = CrossAttentionTimeDecoder(self.latent_dim, self.n, decoder_heads, tcn_dropout)
            self.time_attn_gen = CrossAttentionTimeDecoder(self.latent_dim, self.n, decoder_heads, tcn_dropout)
        else:
            self.time_proj_bus = nn.Linear(temporal_window, self.n)
            self.time_proj_gen = nn.Linear(temporal_window, self.n)


        # ---- Forecast decoders ----
        # Direct multi-step decoding: maps D -> n * F
        # Bus decoder receives z_bus [B, N_bus, D] concatenated with
        # standardized Pd_macro [B, N_bus, 1] -> input dim = D + PD_MACRO_DIM
        exo_bus_dim = len(self.exo_bus_indices) if self.use_exogenous else 0
        exo_gen_d = self.exo_gen_dim if self.use_exogenous else 0

        mlp_hidden_dim = getattr(args.model, "mlp_hidden_dim", 1024)
        mlp_num_layers = getattr(args.model, "mlp_num_layers", 1)

        def build_decoder(in_dim, out_dim):
            layers = []
            curr_dim = in_dim
            for _ in range(mlp_num_layers):
                layers.extend([
                    nn.Linear(curr_dim, mlp_hidden_dim),
                    nn.LayerNorm(mlp_hidden_dim),
                    nn.LeakyReLU(),
                ])
                curr_dim = mlp_hidden_dim
            layers.append(nn.Linear(curr_dim, out_dim))
            return nn.Sequential(*layers)

        # Bus/Gen decoder: latent_dim (+ exo if used)
        self.forecast_decoder_bus = build_decoder(self.latent_dim + exo_bus_dim, self.forecast_bus_dim)
        self.forecast_decoder_gen = build_decoder(
            self.latent_dim + exo_gen_d, self.forecast_gen_dim
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
        # 3. Temporal pass — TCN -> gives sequence
        # ==============================================================
        z_bus_seq = self.tcn_bus(h_bus_4d)  # [B, N_bus, D, W]
        z_gen_seq = self.tcn_gen(h_gen_4d)  # [B, N_gen, D, W]

        if self.use_exogenous:
            pass # (Exogenous features skipped for direct decoding for now)


        # Time decoding: [B, N, D, W] -> [B, N, n, D]
        if self.temporal_decoder == "cross_attention":
            z_bus_trans = self.time_attn_bus(z_bus_seq)
            z_gen_trans = self.time_attn_gen(z_gen_seq)
        else:
            z_bus_trans = self.time_proj_bus(z_bus_seq).permute(0, 1, 3, 2)
            z_gen_trans = self.time_proj_gen(z_gen_seq).permute(0, 1, 3, 2)

        # Feature Projection: [B, N_bus, n, D] -> [B, N_bus, n, F]
        raw_forecast_bus = self.forecast_decoder_bus(z_bus_trans) 
        raw_forecast_gen = self.forecast_decoder_gen(z_gen_trans) 

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
        pg_bounded = bound_with_sigmoid(
            raw_forecast_gen[..., FORECAST_PG_IDX], min_pg, max_pg,
        ).unsqueeze(-1)
        final_forecast_gen = pg_bounded

        vm_bounded = bound_with_sigmoid(
            raw_forecast_bus[..., FORECAST_VM_IDX], min_vm, max_vm,
        ).unsqueeze(-1)
        final_forecast_bus = torch.cat([
            raw_forecast_bus[..., :FORECAST_VM_IDX],
            vm_bounded,
            raw_forecast_bus[..., FORECAST_VM_IDX + 1:],
        ], dim=-1)

        if torch._dynamo.is_compiling():
            return final_forecast_bus, final_forecast_gen
        return {
            "bus": final_forecast_bus,  # [B, N_bus, n, 5]: [Pd, Qd, Qg, Vm, Va]
            "gen": final_forecast_gen,  # [B, N_gen, n, 1]: [Pg]
        }

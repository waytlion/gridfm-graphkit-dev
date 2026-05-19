"""
Standalone Spatio-Temporal GNN for ACOPF forecasting

Contains two classes:
    1. UnmaskedSpatialEncoder - spatial backbone (GNN)
    2. ST_GNN_heterogeneous   - end-to-end forecaster composing spatial encoder,
                                temporal encoder (TCN/transformer) and forecast decoders

The spatial encoder mirrors the TransformerConv layers of GNS_heterogeneous
but strips away masking, physics decoders, and per-layer MLP decoding.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv

from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.utils import bound_with_sigmoid
from gridfm_graphkit.models.temporal_encoders import TCN, TemporalTransformerEncoder
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


class UnmaskedSpatialEncoder(nn.Module):
    """
    spatial backbone mirroring GNS_heterogeneous but stripped
    of all physics decoders, per-layer MLPs, and masking logic.

    Architecture (matches GNS_heterogeneous conv layers):
        - input_proj_{bus,gen,edge}: 2-layer MLP + LayerNorm
        - HeteroConv(TransformerConv) per relation, with skip connections
        - LayerNorm per node type per layer

    Input: x_dict (all features known for historical timesteps)
    Output: latent embeddings h_bus and h_gen
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
        self.alpha_init_res = getattr(args.model, "alpha_init_res", 0.0)# default 0 == no initial residuals
        self.latent_dim = self.hidden_dim * self.heads 

        # --- Input proj ---    
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

        # --- HeteroConv layers  ---
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
        Spatial message-passing on fully observable heterogeneous graphs (No masking, no physics, no MLP decoding)

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

        # for initial residual implementation later on
        h0_bus = None
        h0_gen = None

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

            # Skip connection (first layer changes dim: hidden_dim -> hidden_dim * heads)
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # Initialize H^(0) after first layer for initial residuals
            if h0_bus is None:
                h0_bus = h_bus
                h0_gen = h_gen
            # initial residuals: inject H^(0)
            if self.alpha_init_res != 0.0:
                h_bus = (1.0 - self.alpha_init_res) * h_bus + self.alpha_init_res * h0_bus
                h_gen = (1.0 - self.alpha_init_res) * h_gen + self.alpha_init_res * h0_gen
        
        return h_bus, h_gen


class CrossAttentionTimeDecoder(nn.Module):
    """
    Temporal Decoder: Maps input window W to prediction horizon n
        - Input: [B, N, D, W] 
        - return: [B, N, n, D]
    
    #! Notes:
        - TCN Base model performed worse with this decoder than with the linear decoder on all metrics 

    Architecture:
        - learnable-query cross-attention
        - Query = learnable horizon positional embeddings (E_future) shifted by a context vector, which is the terminal TCN state
            - E_future: Learnable embeddings for each horizon step: [1, n, D] (shortcut to encode horizon-dependent structure early in training - mby useless but prob not a problem)
            - ontext_proj gives sample-specific conditioning (what this particular sequence implies) -> only useful for TCN i believe: context == last TCN state
    """
    def __init__(self, latent_dim: int, horizon: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.E_future = nn.Parameter(torch.empty(1, horizon, latent_dim))
        nn.init.trunc_normal_(self.E_future, std=0.02)
        self.context_proj = nn.Linear(latent_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(latent_dim) 

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """ z_seq: input from temporal encoder
        """
        B, N, D, W = z_seq.shape 
        kv = z_seq.permute(0, 1, 3, 2).contiguous().reshape(B * N, W, D)  
        terminal_state = z_seq[:, :, :, -1].contiguous().reshape(B * N, D) 
        context = self.context_proj(terminal_state).unsqueeze(1) # [B*N, n, D]
        q = self.E_future.expand(B * N, -1, -1) + context # q_base: [1, n, D] -> expand [B*N, n, D]
        attn_out, _ = self.cross_attn(self.norm_q(q), kv, kv)  # Pre-LN cross-attention
        out = q + attn_out                                       # [B*N, n, D]
        return out.reshape(B, N, self.horizon, D)


@MODELS_REGISTRY.register("ST_GNN_heterogeneous")
class ST_GNN_heterogeneous(nn.Module):
    """
    Space-then-Time heterogeneous GNN for MP-ACOPF forecasting

    End-to-end model: predicts exogenous loads AND optimal dispatch
    at the target horizon

    Bus output (5 features): [Pd, Qd, Qg, Vm, Va]
    Gen output (1 feature):  [Pg]

    Pipeline:
        folded_batch [B*W graphs]
          -> UnmaskedSpatialEncoder              -> h_bus, h_gen
          -> unfold                              -> [B, N, W, D]
          -> Temporal encoder (TCN/transformer)  -> z_bus_seq, z_gen_seq  [B, N, D, W]
          -> Time Projection (Linear/Attn)       -> z_bus_trans, z_gen_trans [B, N, n, D]
          -> Condition (step embeddings)         -> z_bus_cond [B, N, n, D_cond]
          -> forecast MLP (per step|shared)      -> y_bus [B, N, n, 5], y_gen [B, N, n, 1]
          -> bound Vm, Pg per step               -> final forecast
    
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
        self.step_embed_dim = getattr(args.model, "step_embed_dim", 0)
        self.temporal_decoder = getattr(args.model, "temporal_decoder", "linear")
        self.forecast_decoder_mode = getattr(args.model, "forecast_decoder_mode", "shared")

        # Step embeddings default to 0 -> should not break the model
        self.step_embeddings = None
        if self.step_embed_dim > 0:
            self.step_embeddings = nn.Embedding(self.n, self.step_embed_dim)
        # ---- Spatial encoder ----
        self.spatial_encoder = UnmaskedSpatialEncoder(args)

        # ---- Dimensions ----
        hidden_dim = args.model.hidden_size
        heads = args.model.attention_head
        self.latent_dim = hidden_dim * heads

        self.forecast_bus_dim = args.model.output_bus_dim  # [Pd, Qd, Qg, Vm, Va]
        self.forecast_gen_dim = args.model.output_gen_dim  # [Pg]

        # ---- Temporal encoders (pluggable: tcn | transformer) ----
        temporal_window = args.data.temporal_window
        temporal_dropout = getattr(args.model, "dropout", 0.0)
        self.temporal_encoder_type = getattr(args.model, "temporal_encoder", "tcn")

        if self.temporal_encoder_type == "tcn":
            tcn_kernel = getattr(args.model, "tcn_kernel_size", 3)
            self.temporal_bus = TCN(
                input_dim=self.latent_dim,
                window_size=temporal_window,
                kernel_size=tcn_kernel,
                dropout=temporal_dropout,
            )
            self.temporal_gen = TCN(
                input_dim=self.latent_dim,
                window_size=temporal_window,
                kernel_size=tcn_kernel,
                dropout=temporal_dropout,
            )
        elif self.temporal_encoder_type == "transformer":
            t_layers = getattr(args.model, "temporal_num_layers", 4)
            t_heads = getattr(args.model, "temporal_num_heads", 8)
            t_ff = 4 * self.latent_dim
            self.temporal_bus = TemporalTransformerEncoder(
                input_dim=self.latent_dim,
                window_size=temporal_window,
                num_layers=t_layers,
                num_heads=t_heads,
                dim_feedforward=t_ff,
                dropout=temporal_dropout,
            )
            self.temporal_gen = TemporalTransformerEncoder(
                input_dim=self.latent_dim,
                window_size=temporal_window,
                num_layers=t_layers,
                num_heads=t_heads,
                dim_feedforward=t_ff,
                dropout=temporal_dropout,
            )
        else:
            raise ValueError(
                f"Unknown temporal_encoder type: '{self.temporal_encoder_type}'. "
                f"Must be 'tcn' or 'transformer'."
            )

        # ---- Temporal decoder ----
        if self.temporal_decoder == "cross_attention":
            decoder_heads = args.model.temporal_decoder_heads
            assert self.latent_dim % decoder_heads == 0, (
                f"latent_dim ({self.latent_dim}) must be divisible by "
                f"temporal_decoder_heads ({decoder_heads})"
            )
            self.time_attn_bus = CrossAttentionTimeDecoder(self.latent_dim, self.n, decoder_heads, temporal_dropout)
            self.time_attn_gen = CrossAttentionTimeDecoder(self.latent_dim, self.n, decoder_heads, temporal_dropout)
        else:
            self.time_proj_bus = nn.Linear(temporal_window, self.n)
            self.time_proj_gen = nn.Linear(temporal_window, self.n)


        # ---- Forecast decoders (shared vs per-horizon) ----
        # Feature decoding: maps D -> F (shared across steps or per-horizon heads)
        exo_bus_dim = len(self.exo_bus_indices) if self.use_exogenous else 0
        exo_gen_d = self.exo_gen_dim if self.use_exogenous else 0

        mlp_hidden_dim = args.model.mlp_hidden_dim
        mlp_num_layers = args.model.mlp_num_layers

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

        bus_decoder_in = self.latent_dim + self.step_embed_dim + exo_bus_dim
        gen_decoder_in = self.latent_dim + self.step_embed_dim + exo_gen_d

        if self.forecast_decoder_mode == "per_horizon":
            # Instantiate n independent MLPs
            self.forecast_decoders_bus = nn.ModuleList([
                build_decoder(bus_decoder_in, self.forecast_bus_dim) for _ in range(self.n)
            ])
            self.forecast_decoders_gen = nn.ModuleList([
                build_decoder(gen_decoder_in, self.forecast_gen_dim) for _ in range(self.n)
            ])
        else:
            # Shared MLP across all horizon steps
            self.forecast_decoder_bus = build_decoder(bus_decoder_in, self.forecast_bus_dim)
            self.forecast_decoder_gen = build_decoder(gen_decoder_in, self.forecast_gen_dim)

    def forward(self, folded_batch, target_batch, B, W, N_bus, one_step: bool = False, step_index: int = 0):
        """
        folded_batch: PyG Batch of B*W window graphs
        target_batch: PyG Batch of B*n target graphs 
        B: batch size
        W: temporal window size
        N_bus: number of bus nodes per graph

        Returns:
            dict with:
                "bus": [B, N_bus, n, 5]  forecast [Pd, Qd, Qg, Vm, Va]
                "gen": [B, N_gen, n, 1]  forecast [Pg]
        """
        D = self.latent_dim
        n = 1 if one_step else self.n

        # spatial pass
        h_bus, h_gen = self.spatial_encoder(
            folded_batch.x_dict,
            folded_batch.edge_index_dict,
            folded_batch.edge_attr_dict,
        ) # h_bus: [B*W*N_bus, D]    h_gen: [B*W*N_gen, D]

        N_gen = h_gen.size(0) // (B * W)

        # unfold
        h_bus_4d = h_bus.view(B, W, N_bus, D).permute(0, 2, 1, 3)  # [B, N_bus, W, D]
        h_gen_4d = h_gen.view(B, W, N_gen, D).permute(0, 2, 1, 3)  # [B, N_gen, W, D]

        # temporal encoder 
        z_bus_seq = self.temporal_bus(h_bus_4d)  # [B, N_bus, D, W]
        z_gen_seq = self.temporal_gen(h_gen_4d)  # [B, N_gen, D, W]

        # temporal decoder [B, N, D, W] -> [B, N, n, D]
        if self.temporal_decoder == "cross_attention":
            z_bus_trans = self.time_attn_bus(z_bus_seq)
            z_gen_trans = self.time_attn_gen(z_gen_seq)
        else:
            z_bus_trans = self.time_proj_bus(z_bus_seq).permute(0, 1, 3, 2)
            z_gen_trans = self.time_proj_gen(z_gen_seq).permute(0, 1, 3, 2)

        if one_step:
            z_bus_trans = z_bus_trans[:, :, :1, :]
            z_gen_trans = z_gen_trans[:, :, :1, :]

        if self.step_embed_dim > 0:
            if one_step:
                step_idx = torch.tensor([step_index], device=z_bus_trans.device)
            else:
                step_idx = torch.arange(n, device=z_bus_trans.device)
            step_emb = self.step_embeddings(step_idx)
            step_emb_bus = step_emb.unsqueeze(0).unsqueeze(0).expand(B, N_bus, n, self.step_embed_dim)
            step_emb_gen = step_emb.unsqueeze(0).unsqueeze(0).expand(B, N_gen, n, self.step_embed_dim)
        else:
            step_emb_bus = z_bus_trans.new_zeros((B, N_bus, n, 0))
            step_emb_gen = z_gen_trans.new_zeros((B, N_gen, n, 0))

        z_bus_cond = torch.cat([z_bus_trans, step_emb_bus], dim=-1)
        z_gen_cond = torch.cat([z_gen_trans, step_emb_gen], dim=-1)

        # forecast decoder: per-horizon heads or shared head across horizons
        if self.forecast_decoder_mode == "per_horizon":
            if one_step:
                raw_forecast_bus = self.forecast_decoders_bus[0](
                    z_bus_cond[:, :, 0, :]
                ).unsqueeze(2)
                raw_forecast_gen = self.forecast_decoders_gen[0](
                    z_gen_cond[:, :, 0, :]
                ).unsqueeze(2)
            else:
                raw_forecast_bus = torch.stack([
                    self.forecast_decoders_bus[k](z_bus_cond[:, :, k, :]) for k in range(n)
                ], dim=2)  # [B, N_bus, n, 5]

                raw_forecast_gen = torch.stack([
                    self.forecast_decoders_gen[k](z_gen_cond[:, :, k, :]) for k in range(n)
                ], dim=2)  # [B, N_gen, n, 1]
        else:
            raw_forecast_bus = self.forecast_decoder_bus(
                z_bus_cond.reshape(B * N_bus * n, -1)
            ).reshape(B, N_bus, n, -1)
            raw_forecast_gen = self.forecast_decoder_gen(
                z_gen_cond.reshape(B * N_gen * n, -1)
            ).reshape(B, N_gen, n, -1)

        # bounds per forecast horizon step: Vm (sigmoid) and Pg (sigmoid) 
        n_target = target_batch["bus"].x.size(0) // (B * N_bus)
        if n_target < 1:
            raise ValueError("target_batch must include at least one forecast step.")
        if one_step and (step_index < 0 or step_index >= n_target):
            raise ValueError(
                f"step_index ({step_index}) must be in [0, {n_target - 1}] for one_step mode."
            )

        target_bus_x_full = target_batch["bus"].x.view(B, n_target, N_bus, -1).permute(0, 2, 1, 3)
        target_gen_x_full = target_batch["gen"].x.view(B, n_target, N_gen, -1).permute(0, 2, 1, 3)
        min_vm = target_bus_x_full[..., MIN_VM_H]  # [B, N_bus, n]
        max_vm = target_bus_x_full[..., MAX_VM_H]
        min_pg = target_gen_x_full[..., MIN_PG]    # [B, N_gen, n]
        max_pg = target_gen_x_full[..., MAX_PG]

        if one_step:
            min_vm = min_vm[:, :, step_index:step_index + 1]
            max_vm = max_vm[:, :, step_index:step_index + 1]
            min_pg = min_pg[:, :, step_index:step_index + 1]
            max_pg = max_pg[:, :, step_index:step_index + 1]

        ## out-of-place ops (torch.cat / reassignment) instead of in-place index
        ## assignment because in place leads to -> runtimeError (due to pytorch precompile)
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

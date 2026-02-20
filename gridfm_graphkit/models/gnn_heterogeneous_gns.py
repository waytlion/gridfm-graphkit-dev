import torch
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.io.param_handler import get_physics_decoder
from torch_scatter import scatter_add
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
    bound_with_sigmoid,
)
from gridfm_graphkit.datasets.globals import (
    # Bus feature indices
    PD_H,      # added for Forecasting Task
    QD_H,      # added for Forecasting Task
    QG_H,      # added for Forecasting Task
    VM_H,
    VA_H,
    MIN_VM_H,
    MAX_VM_H,
    # Output feature indices
    VM_OUT,
    PG_OUT_GEN,
    # Generator feature indices
    PG_H,
    MIN_PG,
    MAX_PG,
)


@MODELS_REGISTRY.register("GNS_heterogeneous")
class GNS_heterogeneous(nn.Module):
    """
    Heterogeneous version of your Transformer-based GNN for buses and generators.
    - Expects node features as dict: x_dict = {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
    - Expects edge_index_dict and edge_attr_dict with keys:
        ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
      (edge_attr only needed for bus-bus currently; other relations can be None)
    - Keeps the physics residual idea but splits it into bus-step and gen-step residuals.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = args.model.num_layers
        self.hidden_dim = args.model.hidden_size
        self.input_bus_dim = args.model.input_bus_dim
        self.input_gen_dim = args.model.input_gen_dim
        self.output_bus_dim = args.model.output_bus_dim
        self.output_gen_dim = args.model.output_gen_dim
        self.edge_dim = args.model.edge_dim
        self.heads = args.model.attention_head
        self.task = args.task.task_name
        self.dropout = getattr(args.model, "dropout", 0.0)
        
        #! What features to predict at t+1. This is basically doing the same filtering as done in AddOPFForecastingMask(). I guess this redundance serves as defensive programming .. but mby double check with Alban
        if self.task == "ForecastOPF":
            # Predict all dynamic features
            self.bus_output_indices = [PD_H, QD_H, QG_H, VM_H, VA_H]  
            self.gen_output_indices = [PG_H]  
    
        else:  # OPF, PowerFlow, StateEstimation
            self.bus_output_indices = [VM_H, VA_H]  
            self.gen_output_indices = [PG_H]  
    


        # projections for each node type
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

        # a small physics MLP that will take residuals (real, imag) and return a correction
        self.physics_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim * self.heads),
            nn.LeakyReLU(),
        )

        # Build hetero layers: HeteroConv of TransformerConv per relation
        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        for i in range(self.num_layers):
            # in-channels depend on whether it is first layer (hidden_dim) or subsequent (hidden_dim * heads)
            in_bus = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            in_gen = self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            out_dim = self.hidden_dim  # TransformerConv will output hidden_dim (per head reduction in HeteroConv call)

            # relation -> conv module mapping
            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }

            hetero_conv = HeteroConv(conv_dict, aggr="sum")
            self.layers.append(hetero_conv)

            # Norms for node representations (note: after HeteroConv each node type will have size out_dim * heads)
            self.norms_bus.append(nn.LayerNorm(out_dim * self.heads))
            self.norms_gen.append(nn.LayerNorm(out_dim * self.heads))

        # Separate shared MLPs to produce final bus/gen outputs (predictions y)
        self.mlp_bus = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_bus_dim),
        )

        self.mlp_gen = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_gen_dim),
        )

        # mask param (kept similar to your original)
        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)

        # container for monitoring residual norms per layer and type
        self.layer_residuals = {}

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, mask_dict):
        """
        x_dict: {"bus": Tensor[num_bus, bus_feat], "gen": Tensor[num_gen, gen_feat]}
        edge_index_dict: keys like ("bus","connects","bus"), ("gen","connected_to","bus"), ("bus","connected_to","gen")
        edge_attr_dict: same keys -> edge attributes (bus-bus requires G,B)
        batch_dict: dict mapping node types to batch tensors (if using batching). Not used heavily here but kept for API parity.
        mask: optional mask per node (applies when computing residuals)
        """

        self.layer_residuals = {}

        # 1) initial projections
        h_bus = self.input_proj_bus(x_dict["bus"])  # [num_bus, hidden_dim]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [num_gen, hidden_dim]

        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        edge_attr_proj_dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        bus_mask = mask_dict["bus"][:, self.bus_output_indices]# [num_bus, len(self.bus_output_indices)] - which features to predict
        gen_mask = mask_dict["gen"][:, self.gen_output_indices]  # [num_gen, len(self.gen_output_indices)] - which features to predict
        bus_fixed = x_dict["bus"][:, self.bus_output_indices]# Ground truth values
        gen_fixed = x_dict["gen"][:, self.gen_output_indices]# Ground truth values

        # iterate layers
        for i, conv in enumerate(self.layers):
            out_dict = conv(
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj_dict,
            )
            out_bus = out_dict["bus"]  # [Nb, hidden_dim * heads]
            out_gen = out_dict["gen"]  # [Ng, hidden_dim * heads]

            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # skip connection
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # Decode bus and generator predictions
            #! Regardless of mask vals
            bus_temp = self.mlp_bus(h_bus)  #! for non forecasting task : [Nb, 2]  -> Vm, Va
            gen_temp = self.mlp_gen(h_gen)  #! for non forecasting : [Ng, 1]  -> Pg

            if self.task == "StateEstimation":
                if i == self.num_layers - 1:
                    Pft, Qft = self.branch_flow_layer(
                        bus_temp,
                        bus_edge_index,
                        bus_edge_attr,
                    )
                    P_in, Q_in = self.node_injection_layer(
                        Pft,
                        Qft,
                        bus_edge_index,
                        num_bus,
                    )
                    output_temp = self.physics_decoder(
                        P_in,
                        Q_in,
                        bus_temp,
                        x_dict["bus"],
                        None,
                        None,
                    )

            else:  # Non-SE tasks
                #! If mask vals = True -> Use predicted values; else use fixed (ground truth) values for physics calculations 
                bus_temp = torch.where(bus_mask, bus_temp, bus_fixed)
                gen_temp = torch.where(gen_mask, gen_temp, gen_fixed)

                # Apply task-specific bounds and prepare input for physics decoder
                if self.task == "ForecastOPF":
                    # Apply sigmoid bounds to voltages (at indices 3, 4 in 5-feature output)
                    bus_temp[:, 3] = bound_with_sigmoid(
                        bus_temp[:, 3],
                        x_dict["bus"][:, MIN_VM_H],
                        x_dict["bus"][:, MAX_VM_H],
                    )
                    gen_temp[:, PG_OUT_GEN] = bound_with_sigmoid(
                        gen_temp[:, PG_OUT_GEN],
                        x_dict["gen"][:, MIN_PG],
                        x_dict["gen"][:, MAX_PG],
                    )
                    
                    #!reorder bus features: physics decoder expects [Vm, Va] at indices [0, 1]
                    bus_temp_reordered = torch.stack([
                        bus_temp[:, 3],  
                        bus_temp[:, 4], 
                    ], dim=1)
                    
                elif self.task == "OptimalPowerFlow":
                    bus_temp[:, VM_OUT] = bound_with_sigmoid(
                        bus_temp[:, VM_OUT],
                        x_dict["bus"][:, MIN_VM_H],
                        x_dict["bus"][:, MAX_VM_H],
                    )
                    
                    gen_temp[:, PG_OUT_GEN] = bound_with_sigmoid(
                        gen_temp[:, PG_OUT_GEN],
                        x_dict["gen"][:, MIN_PG],
                        x_dict["gen"][:, MAX_PG],
                    )
                    
                    bus_temp_reordered = bus_temp  # Already in [Vm, Va] format
                    
                else:  # PowerFlow and other tasks
                    bus_temp_reordered = bus_temp  # Already in [Vm, Va] format
                
                # Compute power flows from voltages (shared for all non-SE tasks)
                Pft, Qft = self.branch_flow_layer(
                    bus_temp_reordered,
                    bus_edge_index,
                    bus_edge_attr,
                )
                P_in, Q_in = self.node_injection_layer(
                    Pft,
                    Qft,
                    bus_edge_index,
                    num_bus,
                )
                agg_bus = scatter_add(
                    gen_temp.squeeze(),
                    gen_to_bus_index,
                    dim=0,
                    dim_size=num_bus,
                )
                
                # Physics decoder validates/derives values from power balance
                physics_output = self.physics_decoder(
                    P_in,
                    Q_in,
                    bus_temp_reordered,  # Input: [Vm, Va] in expected format
                    x_dict["bus"],
                    agg_bus,
                    mask_dict,
                )
                
                # Handle output based on task type
                if self.task == "ForecastOPF":
                    # Keep model predictions for all 5 features (don't override with physics)
                    output_temp = bus_temp.clone()
                else:
                    # Use physics decoder output (overrides predictions with physics-derived values)
                    output_temp = physics_output
                
                # Compute residuals for physics-informed training (shared for all tasks)
                residual_P, residual_Q = self.node_residuals_layer(
                    P_in,
                    Q_in,
                    physics_output,
                    x_dict["bus"],
                )

                bus_residuals = torch.stack([residual_P, residual_Q], dim=-1)

                # Save and project residuals to latent space
                self.layer_residuals[i] = torch.linalg.norm(
                    bus_residuals,
                    dim=-1,
                ).mean()
                h_bus = h_bus + self.physics_mlp(bus_residuals)
        return {"bus": output_temp, "gen": gen_temp}

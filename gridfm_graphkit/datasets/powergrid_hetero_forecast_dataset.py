from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
import os.path as osp
from tqdm import tqdm
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from gridfm_graphkit.datasets.globals import VA_H, PG_H

class HeteroGridForecastDatasetDisk(HeteroGridDatasetDisk):
    """
    One-step-ahead forecasting dataset.
    
    Inherits all processing from HeteroGridDatasetDisk.
    Only difference: y is from t+1 instead of t.
    
    Input:  x_dict at time t
    Target: y_dict at time t+1
    """

    def process(self):
        print("LOADING DATA")
        bus_data = pd.read_parquet(osp.join(self.raw_dir, "bus_data.parquet"))
        gen_data = pd.read_parquet(osp.join(self.raw_dir, "gen_data.parquet"))
        branch_data = pd.read_parquet(osp.join(self.raw_dir, "branch_data.parquet"))

        assert bus_data.scenario.min() == 0 and bus_data.scenario.max() == len(bus_data.scenario.unique()) - 1

        load_scenarios = torch.tensor(
            bus_data.groupby("scenario")["load_scenario_idx"].first().values,
        )
        torch.save(load_scenarios, osp.join(self.processed_dir, "load_scenarios.pt"))

        agg_gen = (
            gen_data.groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]]
            .sum()
            .reset_index()
        )
        bus_data = bus_data.merge(agg_gen, on=["scenario", "bus"], how="left").fillna(0)

        done_path = osp.join(self.processed_dir, self.processed_done_file)
        if osp.exists(done_path):
            print("Processed files already exist. Skipping processing.")
            return
        
        bus_features = self.BUS_FEATURES
        gen_features = self.GEN_FEATURES
        forward_branch_features = self.FORWARD_BRANCH_FEATURES
        reverse_branch_features = self.REVERSE_BRANCH_FEATURES

        # Group by scenario
        bus_groups = bus_data.groupby("scenario")
        gen_groups = gen_data.groupby("scenario")
        branch_groups = branch_data.groupby("scenario")

        #! SORT
        scenarios_sorted = sorted(bus_data["scenario"].unique())
        # Process each scenario
        for i, scenario in enumerate(tqdm(scenarios_sorted[:-1], desc="Processing forecast pairs")):
            if (
                scenario not in gen_groups.groups
                or scenario not in branch_groups.groups
            ):
                raise ValueError

            #! GET NEXT SCENARIO FOR FORECASTING TARGET
            next_scenario = scenarios_sorted[i + 1]  

            data = HeteroData()

            # #! timestep t 
            bus_df = bus_groups.get_group(scenario).reset_index()
            data["bus"].x = torch.tensor(bus_df[bus_features].values, dtype=torch.float)
            gen_df = gen_groups.get_group(scenario).reset_index()
            data["gen"].x = torch.tensor(gen_df[gen_features].values, dtype=torch.float)
            gen_df["gen_index"] = gen_df.index  # Use actual index as generator ID

            #! timestep t+1
            next_bus_df = bus_groups.get_group(next_scenario).reset_index()
            data["bus"].y = torch.tensor(
                next_bus_df[bus_features].values[:, : 5],
                dtype=torch.float
            )
            
            next_gen_df = gen_groups.get_group(next_scenario).reset_index()
            data["gen"].y = torch.tensor(
                next_gen_df[gen_features].values[:, : 1],
                dtype=torch.float
            )

            # Bus-Bus edges
            #! ASSUMTION: Graph stays same from t -> t+1 ---> Use branch at timestep t
            branch_df = branch_groups.get_group(scenario)

            forward_edges = torch.tensor(
                branch_df[["from_bus", "to_bus"]].values.T,
                dtype=torch.long,
            )
            forward_edge_attr = torch.tensor(
                branch_df[forward_branch_features].values,
                dtype=torch.float,
            )

            reverse_edges = torch.tensor(
                branch_df[["to_bus", "from_bus"]].values.T,
                dtype=torch.long,
            )
            reverse_edge_attr = torch.tensor(
                branch_df[reverse_branch_features].values,
                dtype=torch.float,
            )

            edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
            edge_attr = torch.cat([forward_edge_attr, reverse_edge_attr], dim=0)

            forward_targets = torch.tensor(
                branch_df[["pf", "qf"]].values,
                dtype=torch.float,
            )
            reverse_targets = torch.tensor(
                branch_df[["pt", "qt"]].values,
                dtype=torch.float,
            )
            edge_y = torch.cat([forward_targets, reverse_targets], dim=0)

            data["bus", "connects", "bus"].edge_index = edge_index
            data["bus", "connects", "bus"].edge_attr = edge_attr
            data["bus", "connects", "bus"].y = edge_y

            # Gen-Bus and Bus-Gen edges
            data["gen", "connected_to", "bus"].edge_index = torch.tensor(
                gen_df[["gen_index", "bus"]].values.T,
                dtype=torch.long,
            )
            data["bus", "connected_to", "gen"].edge_index = torch.tensor(
                gen_df[["bus", "gen_index"]].values.T,
                dtype=torch.long,
            )

            data["scenario_id"] = torch.tensor([scenario], dtype=torch.long)

            # Save graph
            torch.save(
                data.to_dict(),
                osp.join(self.processed_dir, f"data_index_{i}.pt"),
            )

        with open(osp.join(self.processed_dir, self.processed_done_file), "w") as f:
            f.write("done")

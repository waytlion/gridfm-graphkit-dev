import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import torch.distributed as dist
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    load_normalizer,
    get_task_transforms,
)
from gridfm_graphkit.datasets.utils import (
    split_dataset,
    split_dataset_by_load_scenario_idx,
)
import numpy as np
import random
import warnings
import os
import lightning as L
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_forecast_dataset import HeteroGridForecastDatasetDisk

class LitGridHeteroForecastDataModule(LitGridHeteroDataModule):
    """
    DataModule for one-step-ahead forecasting.
    
    Inherits all data loading from LitGridHeteroDataModule.
    Differences:
     1.  Uses HeteroGridForecastDatasetDisk instead of HeteroGridForecastDatasetDisk.
     2. Chronological Data Split
    """
    
    def setup(self, stage: str):
        if self._is_setup_done:
            print(f"Setup already done for stage={stage}, skipping...")
            return

        for i, network in enumerate(self.args.data.networks):
            data_normalizer = load_normalizer(args=self.args)
            self.data_normalizers.append(data_normalizer)

            # Create torch dataset and split
            data_path_network = os.path.join(self.data_dir, network)

            # Run preprocessing only on rank 0
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"Pre-processing of {network} dataset on rank 0")
                _ = HeteroGridForecastDatasetDisk(  # just to trigger processing
                    root=data_path_network,
                    norm_method=self.args.data.normalization,
                    data_normalizer=data_normalizer,
                    transform=get_task_transforms(args=self.args),
                )

            # All ranks wait here until processing is done
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

            dataset = HeteroGridForecastDatasetDisk(
                root=data_path_network,
                norm_method=self.args.data.normalization,
                data_normalizer=data_normalizer,
                transform=get_task_transforms(args=self.args),
            )
            self.datasets.append(dataset)

            num_scenarios = self.args.data.scenarios[i]
            if num_scenarios > len(dataset):
                warnings.warn(
                    f"Requested number of scenarios ({num_scenarios}) exceeds dataset size ({len(dataset)}). "
                    "Using the full dataset instead.",
                )
                num_scenarios = len(dataset)

            # Create a subset
            all_indices = list(range(len(dataset)))
            # Random seed set before every shuffle for reproducibility in case the power grid datasets are analyzed in a different order
            random.seed(self.args.seed)
            random.shuffle(all_indices)
            subset_indices = all_indices[:num_scenarios]

            # load_scenario for each scenario in the subset
            load_scenarios = dataset.load_scenarios[subset_indices]

            dataset = Subset(dataset, subset_indices)

            np.random.seed(self.args.seed)

            #! NEW: Check for temporal forecasting split flag
            if getattr(self.args.data, 'temporal_split', False):
                from gridfm_graphkit.datasets.utils import split_dataset_by_time
                train_dataset, val_dataset, test_dataset = split_dataset_by_time(
                    dataset,
                    self.data_dir,
                    load_scenarios,
                    self.args.data.val_ratio,
                    self.args.data.test_ratio,
                )
            elif self.split_by_load_scenario_idx:
                train_dataset, val_dataset, test_dataset = (
                    split_dataset_by_load_scenario_idx(
                        dataset,
                        self.data_dir,
                        load_scenarios,
                        self.args.data.val_ratio,
                        self.args.data.test_ratio,
                    )
                )
            else:
                train_dataset, val_dataset, test_dataset = split_dataset(
                    dataset,
                    self.data_dir,
                    self.args.data.val_ratio,
                    self.args.data.test_ratio,
                )

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)

        self.train_dataset_multi = ConcatDataset(self.train_datasets)
        self.val_dataset_multi = ConcatDataset(self.val_datasets)
        self._is_setup_done = True
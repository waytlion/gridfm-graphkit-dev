import os
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import Subset
from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets.utils import (
    split_dataset,
    split_dataset_by_load_scenario_idx,
    split_dataset_by_time,
)
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_forecast_dataset import HeteroGridForecastDatasetDisk

class LitGridHeteroForecastDataModule(LitGridHeteroDataModule):
    """
    DataModule for one-step-ahead forecasting.
    
    Inherits all ormalization & setup from LitGridHeteroDataModule.
    Differences:
     1.  Uses HeteroGridForecastDatasetDisk (t → t+1 pairs).
     2.  Supports temporal_split for Chronological Data Split
    """
    
    def _create_dataset(self, data_path_network, data_normalizer):
        """Override to use forecast dataset class."""
        return HeteroGridForecastDatasetDisk(
            root=data_path_network,
            data_normalizer=data_normalizer,
            transform=get_task_transforms(args=self.args),
        )

    def _split_dataset(self, dataset, load_scenarios, val_ratio, test_ratio):
        """
        Override splitting to support temporal_split flag.
        
        Args:
            dataset: Subset to split
            load_scenarios: Load scenario indices for each sample
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Check for temporal forecasting split flag
        if getattr(self.args.data, 'temporal_split', False):
            return split_dataset_by_time(
                dataset,
                self.data_dir,
                load_scenarios,
                val_ratio,
                test_ratio,
            )
        elif self.split_by_load_scenario_idx:
            return split_dataset_by_load_scenario_idx(
                dataset,
                self.data_dir,
                load_scenarios,
                val_ratio,
                test_ratio,
            )
        else:
            return split_dataset(
                dataset,
                self.data_dir,
                val_ratio,
                test_ratio,
            )
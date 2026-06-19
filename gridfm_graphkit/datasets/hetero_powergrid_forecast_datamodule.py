import os
import random
import warnings
import numpy as np
import torch.distributed as dist
from torch.utils.data import Subset
from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_forecast_dataset import HeteroGridForecastDatasetDisk

class LitGridHeteroForecastDataModule(LitGridHeteroDataModule):
    """
    DataModule for one-step-ahead forecasting.
    
    Inherits all normalization & setup from LitGridHeteroDataModule.
    Differences:
     1.  Uses HeteroGridForecastDatasetDisk (t → t+1 pairs).

    Note: chronological splitting via ``temporal_split: true`` is now provided by the base
    ``LitGridHeteroDataModule._split_dataset`` (hoisted there so the OPF surrogate task can
    reuse it). This subclass therefore no longer overrides ``_split_dataset``.
    """

    def _create_dataset(self, data_path_network, data_normalizer):
        """Override to use forecast dataset class."""
        return HeteroGridForecastDatasetDisk(
            root=data_path_network,
            data_normalizer=data_normalizer,
            transform=get_task_transforms(args=self.args),
        )
"""
DataModule for the ST-GNN Spatio-Temporal MP-ACOPF forecasting task.

Inherits from :class:`LitGridHeteroForecastDataModule` to reuse normalizer
fitting and temporal-split logic.  Differences:

1. Uses :class:`HeteroGridTemporalDatasetDisk` (sliding-window over individual
   scenario graphs) instead of forecast pairs.
2. Uses :func:`collate_temporal` with ``torch.utils.data.DataLoader`` instead
   of PyG's ``DataLoader``, producing folded ``(B·W)`` batches.
3. Reads ``window_size`` and ``forecast_horizon`` from ``args.data``.
"""

import os
from torch.utils.data import DataLoader

from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets.hetero_powergrid_forecast_datamodule import (
    LitGridHeteroForecastDataModule,
)
from gridfm_graphkit.datasets.powergrid_hetero_temporal_dataset import (
    HeteroGridTemporalDatasetDisk,
    collate_temporal,
)


class LitGridHeteroTemporalDataModule(LitGridHeteroForecastDataModule):
    """
    DataModule for sliding-window spatio-temporal forecasting.

    Inherits temporal-split support from
    :class:`LitGridHeteroForecastDataModule`.  Overrides dataset creation
    (to use :class:`HeteroGridTemporalDatasetDisk`) and all dataloader
    methods (to use :func:`collate_temporal`).

    Config keys read from ``args.data``:
        - ``temporal_window`` (int): lookback window *W* (required).
        - ``forecast_horizon`` (int, optional): steps ahead *n* (default 1).
    """

    # ------------------------------------------------------------------
    # Dataset creation hook
    # ------------------------------------------------------------------

    def _create_dataset(self, data_path_network, data_normalizer):
        """Override to use temporal-window dataset."""
        window_size = self.args.data.temporal_window
        forecast_horizon = getattr(self.args.data, "forecast_horizon", 1)

        return HeteroGridTemporalDatasetDisk(
            root=data_path_network,
            data_normalizer=data_normalizer,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            transform=get_task_transforms(args=self.args),
        )

    # ------------------------------------------------------------------
    # DataLoaders — use torch DataLoader + collate_temporal
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_multi,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.data.workers,
            pin_memory=True,
            collate_fn=collate_temporal,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_multi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.data.workers,
            pin_memory=True,
            collate_fn=collate_temporal,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
                collate_fn=collate_temporal,
            )
            for ds in self.test_datasets
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
                collate_fn=collate_temporal,
            )
            for ds in self.test_datasets
        ]

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
from functools import partial
from torch.utils.data import DataLoader

from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets.hetero_powergrid_forecast_datamodule import (
    LitGridHeteroForecastDataModule,
)
from gridfm_graphkit.datasets.powergrid_hetero_temporal_dataset import (
    HeteroGridTemporalDatasetDisk,
    build_cyclical_time_table,
    collate_temporal,
    collate_temporal_window_norm,
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

    def _build_time_features_table(self):
        """Build (and cache) the [T, 6] cyclical time-feature table if enabled.

        Toggle ``data.append_time_features`` (default ON). When on, the table is
        sized to the longest scenario timeline across networks and indexed at
        collate time by each graph's chronological ``scenario_id``. Returns None
        when explicitly disabled (``append_time_features: false``).
        """
        if not getattr(self.args.data, "append_time_features", True):
            return None
        if getattr(self, "_time_features_table", None) is None:
            # Longest chronological timeline across networks (scenario_id is a
            # global hour offset from the shared start_date).
            T = max(
                int(getattr(ds, "_total_scenarios", len(ds)))
                for ds in self.datasets
            )
            start_date = getattr(self.args.data, "time_feature_start_date", "2019-01-01")
            frequency = getattr(self.args.data, "time_feature_frequency", "h")
            self._time_features_table = build_cyclical_time_table(
                T, start_date=start_date, frequency=frequency
            )
        return self._time_features_table

    def _get_collate_fn(self):
        """Return the appropriate collate function based on the normalizer type."""
        from gridfm_graphkit.datasets.normalizers import HeteroDataWindowMVANormalizer
        if self.data_normalizers and isinstance(
            self.data_normalizers[0], HeteroDataWindowMVANormalizer
        ):
            base_collate = collate_temporal_window_norm
        else:
            base_collate = collate_temporal

        time_features_table = self._build_time_features_table()
        if time_features_table is not None:
            return partial(base_collate, time_features_table=time_features_table)
        return base_collate

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
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_multi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.data.workers,
            pin_memory=True,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
                collate_fn=self._get_collate_fn(),
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
                collate_fn=self._get_collate_fn(),
            )
            for ds in self.test_datasets
        ]

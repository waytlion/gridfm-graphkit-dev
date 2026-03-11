"""
Temporal Window Dataset for ST-GNN MP-ACOPF Forecasting.

Provides sliding-window access over sequentially-indexed scenario graphs
processed by HeteroGridDatasetDisk. Each individual scenario is stored as
a separate data_index_{scenario}.pt file (individual scenario format).

Each sample returns:
    - window_graphs: list of W consecutive HeteroData graphs for timesteps
      [t-W+1, ..., t], carrying x_dict, y_dict, edge data, and mask_dict.
    - target_graphs: list of n consecutive HeteroData graphs for timesteps
      [t+1, ..., t+n], carrying prediction targets (y_dict) and full
      exogenous features (x_dict, including loads Pd, Qd for late fusion).

- collate_temporal() function is designed so that the folded batch can be unfolded via a single view(B, W, N_bus, D) call, provided N_bus is constant across all graphs.

- Uses a separate processed directory ('processed_temporal/')
"""

import os.path as osp
from typing import List, Tuple, Optional, Callable

import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData, Batch

from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
from gridfm_graphkit.datasets.normalizers import Normalizer


class HeteroGridTemporalDatasetDisk(HeteroGridDatasetDisk):
    """
    Sliding-window temporal dataset built on top of HeteroGridDatasetDisk.

    - Reuses the parent's process() to create individual scenario graphs
    - stores them in 'processed_temporal/
    - __getitem__ returns (window_graphs, target_graphs) instead of a single graph.

    Args:
        root: Root directory (must contain raw/ with parquet files).
        data_normalizer: Normalizer instance (fitted externally by datamodule).
        window_size: Number of historical timesteps W in each window.
        forecast_horizon: Number of steps ahead n (default 1).
            Returns n target graphs for timesteps [t+1, ..., t+n].
        transform: Optional runtime transform (e.g. masking).
        pre_transform: Optional pre-processing transform.
        pre_filter: Optional filter.
    """

    def __init__(
        self,
        root: str,
        data_normalizer: Normalizer,
        window_size: int,
        forecast_horizon: int = 1,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        # Parent __init__ triggers process() if needed, counts files, loads
        # load_scenarios, etc.
        super().__init__(root, data_normalizer, transform, pre_transform, pre_filter)

        # Total individual scenario files available (set by parent's len())
        self._total_scenarios = super().len()

    # ------------------------------------------------------------------
    # Separate processed directory to avoid collisions
    # ------------------------------------------------------------------

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_temporal")

    # ------------------------------------------------------------------
    # Dataset length & access
    # ------------------------------------------------------------------

    def len(self) -> int:
        """Number of valid (window, targets) samples."""
        return max(0, self._total_scenarios - self.window_size - self.forecast_horizon + 1)

    def __getitem__(self, idx: int):
        """
        Return a temporal sample.

        idx: Sample index (0-based).
            Window covers scenarios [idx, idx+W-1].
            Targets cover scenarios [idx+W, idx+W+n-1].

        Returns:
            (window_graphs, target_graphs) where:

            - window_graphs: List[HeteroData] of length W, oldest to newest.
              Each graph has x_dict, y_dict, edge_index_dict, edge_attr_dict.
              mask_dict is added by the transform if one is set.
            - target_graphs: List[HeteroData] of length n, for timesteps
              t+1 through t+n. Each carries x_dict (full 15-dim bus features including
              exogenous Pd, Qd for late fusion) and y_dict (prediction targets).
        """
        if not isinstance(idx, int):
            raise NotImplementedError(
                "HeteroGridTemporalDatasetDisk only supports integer indexing."
            )

        # Map through indices() for Subset compatibility
        actual_idx = self.indices()[idx]

        # --- Build window [actual_idx, ..., actual_idx + W - 1] ---
        window_graphs: List[HeteroData] = []
        for t in range(self.window_size):
            graph = self.get(actual_idx + t)
            window_graphs.append(graph)

        # --- Targets [actual_idx + W, ..., actual_idx + W + n - 1] ---
        target_graphs: List[HeteroData] = []
        for h in range(self.forecast_horizon):
            target_idx = actual_idx + self.window_size + h
            target_graphs.append(self.get(target_idx))

        return window_graphs, target_graphs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Pre-load all individual scenario graphs into RAM.

        Loads all _total_scenarios scenario files, normalizes, and applies
        transforms. __getitem__ assembles windows entirely from the
        in-memory list — zero disk I/O during training.
        Must be called after the normalizer has been fitted.
        """
        self._data_list = []
        for scenario_idx in tqdm(
            range(self._total_scenarios),
            desc=f"Pre-loading {self.__class__.__name__} into RAM",
        ):
            file_name = osp.join(self.processed_dir, f"data_index_{scenario_idx}.pt")
            data_dict = torch.load(file_name, weights_only=True)
            data = HeteroData.from_dict(data_dict)
            self.data_normalizer.transform(data=data)
            if self.transform is not None:
                data = self.transform(data)
            self._data_list.append(data)

        print("Dataset initialized. This should only print ONCE.")
        total_bytes = sum(
            v.nbytes
            for data in self._data_list
            for store in data.node_stores + data.edge_stores
            for v in store.values()
            if isinstance(v, torch.Tensor)
        )
        print(f"  RAM estimate: {total_bytes / 1e9:.2f} GB ({len(self._data_list)} individual scenarios)")

    def get(self, scenario_idx: int) -> HeteroData:
        """Override parent's get() to apply self.transform in the disk-fallback path.

        The temporal __getitem__ calls self.get() directly, bypassing PyG's
        Dataset.__getitem__ wrapper — so transform must be applied here explicitly
        rather than relying on PyG to do it after the call.
        """
        #HPC path -> load all data to system-RAM once
        if self._data_list is not None:
            return self._data_list[scenario_idx]
        file_name = osp.join(
            self.processed_dir, f"data_index_{scenario_idx}.pt",
        )
        if not osp.exists(file_name):
            raise IndexError(
                f"Scenario file {file_name} not found. Ensure the dataset "
                f"was processed (individual scenario format)."
            )

        data_dict = torch.load(file_name, weights_only=True)
        data = HeteroData.from_dict(data_dict)

        # Per-sample normalization (same as parent's get())
        self.data_normalizer.transform(data=data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    # ------------------------------------------------------------------
    # Load-scenario mapping (for the datamodule's temporal split)
    # ------------------------------------------------------------------

    def get_window_load_scenarios(self) -> torch.Tensor:
        """Return load_scenario_idx for each valid window sample.

        For sample idx in [0, len()-1], the last window timestep is idx+W-1.
        -> idx+W-1 ranges from W-1 to S-n-1 (inclusive).
        -> Python slice [W-1 : S-n] gives indices W-1..S-n-1 (right-exclusive).
        """
        return self.load_scenarios[
            self.window_size - 1 : self._total_scenarios - self.forecast_horizon
        ]


# ======================================================================
# Collation
# ======================================================================


def collate_temporal(
    batch_list: List[Tuple[List[HeteroData], List[HeteroData]]],
) -> dict:
    """
    Custom collate function for HeteroGridTemporalDatasetDisk.

    Needed because PyG's default collate expects a flat list of HeteroData,
    but our dataset returns (window_graphs, target_graphs) tuples. This
    function:
      1. Flattens all window graphs into one folded Batch (B*W graphs).
      2. Flattens all target graphs into one target Batch (B*n graphs).
      3. Returns B, W, n, N_bus metadata for vectorized unfolding.

    Memory layout: graphs are ordered sample-major, time-minor so that
    h_bus can be unfolded via:
        h_bus.view(B, W, N_bus, D).permute(0, 2, 1, 3)  -> [B, N_bus, W, D]

    Args:
        batch_list: list of (window_graphs, target_graphs) tuples from the
            DataLoader, one per sample.

    Returns:
        dict with keys:
            folded_batch: Batch of B*W window graphs
            target_batch: Batch of B*n target graphs
            B: batch size
            W: window size
            n: forecast horizon
            N_bus: number of bus nodes per graph
    """
    B = len(batch_list)
    W = len(batch_list[0][0])
    n = len(batch_list[0][1])

    # Flatten: sample-major, time-minor order
    all_window_graphs: List[HeteroData] = []
    all_target_graphs: List[HeteroData] = []

    for window_graphs, target_graphs in batch_list:
        assert len(window_graphs) == W
        assert len(target_graphs) == n
        all_window_graphs.extend(window_graphs)
        all_target_graphs.extend(target_graphs)

    # N_bus from the first graph (constant per network)
    N_bus = all_window_graphs[0]["bus"].x.size(0)

    folded_batch = Batch.from_data_list(all_window_graphs)
    target_batch = Batch.from_data_list(all_target_graphs)

    return {
        "folded_batch": folded_batch,
        "target_batch": target_batch,
        "B": B,
        "W": W,
        "n": n,
        "N_bus": N_bus,
    }



import torch
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk


class HeteroGridTwoStepDataset(HeteroGridDatasetDisk):
    """Two-step OPF dataset (OPF surrogate on forecasted loads).
    - Same as HeteroGridDatasetDisk, plus reads the true realized load into bus.true_load
    - Expects extra bus_data.parquet columns: Pd_true, Qd_true (physical MW/MVar)
    - Consumed by OptimalPowerFlowTwoStepTask (residual scored vs true load)
    """
    def _attach_extra_bus_attrs(self, data, bus_df):
        # true_load = [N_bus, 2] (Pd_true, Qd_true)
        data["bus"].true_load = torch.tensor(
            bus_df[["Pd_true", "Qd_true"]].values, dtype=torch.float
        )

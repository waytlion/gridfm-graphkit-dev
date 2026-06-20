"""Datamodule for the two-step OPF task (OPF surrogate on forecasted loads).

- Uses HeteroGridTwoStepDataset (carries bus.true_load).
- Otherwise identical to LitGridHeteroDataModule (single-snapshot OPF).
"""

from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_twostep_dataset import (
    HeteroGridTwoStepDataset,
)


class LitGridHeteroTwoStepDataModule(LitGridHeteroDataModule):
    def _create_dataset(self, data_path_network, data_normalizer):
        return HeteroGridTwoStepDataset(
            root=data_path_network,
            data_normalizer=data_normalizer,
            transform=get_task_transforms(args=self.args),
        )

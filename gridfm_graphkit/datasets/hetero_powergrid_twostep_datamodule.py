

from gridfm_graphkit.io.param_handler import get_task_transforms
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.powergrid_hetero_twostep_dataset import (
    HeteroGridTwoStepDataset,
)


class LitGridHeteroTwoStepDataModule(LitGridHeteroDataModule):
    """Datamodule for two-step task (OPF surrogate on forecasted loads)

    - identical to LitGridHeteroDataModule (single-snapshot OPF), but using HeteroGridTwoStepDataset (carries bus.true_load)
    """
    def _create_dataset(self, data_path_network, data_normalizer):
        return HeteroGridTwoStepDataset(
            root=data_path_network,
            data_normalizer=data_normalizer,
            transform=get_task_transforms(args=self.args),
        )

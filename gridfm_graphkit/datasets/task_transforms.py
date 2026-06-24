from torch_geometric.transforms import Compose
from gridfm_graphkit.datasets.transforms import (
    RemoveInactiveBranches,
    RemoveInactiveGenerators,
    ApplyMasking,
    LoadGridParamsFromPath,
)
from gridfm_graphkit.datasets.masking import (
    AddOPFHeteroMask,
    AddPFHeteroMask,
    AddOPFForecastingMask,
    SimulateMeasurements,
)
from gridfm_graphkit.io.registries import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register("PowerFlow")
class PowerFlowTransforms(Compose):
    def __init__(self, args):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(AddPFHeteroMask())
        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)


@TRANSFORM_REGISTRY.register("OptimalPowerFlow")
class OptimalPowerFlowTransforms(Compose):
    def __init__(self, args):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(AddOPFHeteroMask())
        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)


@TRANSFORM_REGISTRY.register("OptimalPowerFlowTwoStep")
class OptimalPowerFlowTwoStepTransforms(OptimalPowerFlowTransforms):
    pass


@TRANSFORM_REGISTRY.register("ForecastOPF")
class ForecastOPFTransforms(Compose):
    """
    Transform for forecast OPF task - no feature masking.
    All features visible at time t, all features predicted at time t+1.
    """
    
    def __init__(self, args):
        transforms = []

        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(AddOPFForecastingMask()) # Tells model which features to predict (they get set to 1/true)
        #! No ApplyMasking(), because ApplyMasking() determines, which input data the model sees -> If we were to do apply Masking, the model would not see all input features
        # transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)

@TRANSFORM_REGISTRY.register("ST_ForecastOPF")
class ST_ForecastOPFTransforms(Compose):
    """
    Transform for ST forecast OPF task - identical to ForecastOPF.
    All features visible at time t, all features predicted at t+1..t+n.
    """
    def __init__(self, args):
        transforms = []
        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(AddOPFForecastingMask())
        # No ApplyMasking — same rationale as ForecastOPF
        super().__init__(transforms)

@TRANSFORM_REGISTRY.register("StateEstimation")
class StateEstimationTransforms(Compose):
    def __init__(self, args):
        transforms = []

        if hasattr(args.task, "grid_path"):
            transforms.append(LoadGridParamsFromPath(args))
        transforms.append(RemoveInactiveBranches())
        transforms.append(RemoveInactiveGenerators())
        transforms.append(SimulateMeasurements(args=args))
        transforms.append(ApplyMasking(args=args))

        # Pass the list of transforms to Compose
        super().__init__(transforms)

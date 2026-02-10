# diagnostic.py
from gridfm_graphkit.datasets.powergrid_hetero_forecast_dataset import HeteroGridForecastDatasetDisk
from gridfm_graphkit.tasks.task_transforms import get_task_transforms
from gridfm_graphkit.io.param_handler import load_config

# Load data without transform
dataset_raw = HeteroGridForecastDatasetDisk(
    root="./data/case14_ieee",
    norm_method="none",
    data_normalizer=None,
    transform=None
)
data_raw = dataset_raw[0]

# Load with transform
args = load_config("examples/config/HGNS_OPF_datakit_case14.yaml")
transform = get_task_transforms(args)
data_transformed = transform(data_raw)

print("=== DIAGNOSTIC ===")
print(f"Raw y shape:         {data_raw['bus'].y.shape}")
print(f"Transformed y shape: {data_transformed['bus'].y.shape}")
print(f"Raw y[0]:            {data_raw['bus'].y[0]}")
print(f"Transformed y[0]:    {data_transformed['bus'].y[0]}")

if data_raw['bus'].y.shape != data_transformed['bus'].y.shape:
    print("\n⚠️ PROBLEM: Transform is slicing y!")
    print("   → Need to modify task_transforms.py")
else:
    print("\n✅ No slicing detected, just change output_bus_dim")
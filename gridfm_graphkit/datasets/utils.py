import numpy as np
from torch.utils.data import Subset
from typing import Tuple
from torch import Tensor
import torch


def split_dataset(
    dataset,
    log_dir: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[Subset, Subset, Subset]:
    """
    Splits a dataset into training, validation, and test sets, and logs the indices for each split to CSV files for further analysis

    Args:
        dataset (torch_geometric.dataDataset): The dataset to split.
        log_dir (str): Directory where CSV files containing the indices for each split will be saved.
        val_ratio (float, optional): Proportion of the dataset to include in the validation set.
        test_ratio (float, optional): Proportion of the dataset to include in the test set.

    Raises:
        ValueError: If `val_ratio + test_ratio >= 1`, which would leave no data for the training set.

    Returns:
        tuple: A tuple containing:
            - train_dataset (torch.utils.data.Subset): The training subset of the dataset.
            - val_dataset (torch.utils.data.Subset): The validation subset of the dataset.
            - test_dataset (torch.utils.data.Subset): The test subset of the dataset.
    """

    if val_ratio + test_ratio >= 1:
        raise ValueError("The sum of val_ratio and test_ratio must be less than 1.")

    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - val_size - test_size

    # Generate shuffled indices and split manually
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def split_dataset_by_load_scenario_idx(
    dataset,
    log_dir: str,
    load_scenarios: Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[Subset, Subset, Subset]:
    if val_ratio + test_ratio >= 1:
        raise ValueError("The sum of val_ratio and test_ratio must be less than 1.")

    unique_load_scenarios = torch.unique(load_scenarios)
    val_size = int(val_ratio * len(unique_load_scenarios))
    test_size = int(test_ratio * len(unique_load_scenarios))
    train_size = len(unique_load_scenarios) - val_size - test_size

    unique_load_scenarios = torch.tensor(np.random.permutation(unique_load_scenarios))
    train_load_scenarios = unique_load_scenarios[:train_size]
    val_load_scenarios = unique_load_scenarios[train_size : train_size + val_size]
    test_load_scenarios = unique_load_scenarios[train_size + val_size :]

    train_indices = torch.nonzero(torch.isin(load_scenarios, train_load_scenarios)).flatten().tolist()
    val_indices = torch.nonzero(torch.isin(load_scenarios, val_load_scenarios)).flatten().tolist()
    test_indices = torch.nonzero(torch.isin(load_scenarios, test_load_scenarios)).flatten().tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def split_dataset_by_time(
    dataset,
    log_dir: str,
    load_scenarios: Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_window: int = None,
) -> Tuple[Subset, Subset, Subset]:
    """
    This function is ALMOST identical to <split_dataset_by_load_scenario_idx>
    -> DIFFERENCE: Split dataset CHRONOLOGICALLY for forecasting tasks.
    
    Args:
        dataset: PyG Dataset
        log_dir: Where to save split indices
        load_scenarios: Tensor of scenario IDs (e.g., [0,0,0,...,1,1,1,...,364,364])
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for test (default 0.1)
    
    Returns:
        (train_dataset, val_dataset, test_dataset) as Subsets
    """
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")
    
    #! unique scenarios in CHRONOLOGICAL order
    unique_load_scenarios = torch.unique(load_scenarios, sorted=True)  
    
    # calc split sizes
    n_scenarios = len(unique_load_scenarios)
    val_size = int(val_ratio * n_scenarios)
    test_size = int(test_ratio * n_scenarios)
    train_size = n_scenarios - val_size - test_size
    
    # Split scenarios chronologically
    train_load_scenarios = unique_load_scenarios[:train_size]
    # Optional: cap training to the most-recent `train_window` scenarios (drop older
    # history) while keeping val/test IDENTICAL. Enables comparable-test-set studies
    # (e.g. 3yr vs 24yr training, same held-out future test window).
    if train_window is not None and 0 < train_window < len(train_load_scenarios):
        train_load_scenarios = train_load_scenarios[-train_window:]
    val_load_scenarios = unique_load_scenarios[train_size:train_size + val_size]
    test_load_scenarios = unique_load_scenarios[train_size + val_size:]
    
    # Get indices for each split
    train_indices = torch.nonzero(torch.isin(load_scenarios, train_load_scenarios)).squeeze()
    val_indices = torch.nonzero(torch.isin(load_scenarios, val_load_scenarios)).squeeze()
    test_indices = torch.nonzero(torch.isin(load_scenarios, test_load_scenarios)).squeeze()
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Temporal split:")
    print(f"  Train: scenarios {train_load_scenarios[0]}-{train_load_scenarios[-1]} ({len(train_indices)} samples)")
    print(f"  Val:   scenarios {val_load_scenarios[0]}-{val_load_scenarios[-1]} ({len(val_indices)} samples)")
    print(f"  Test:  scenarios {test_load_scenarios[0]}-{test_load_scenarios[-1]} ({len(test_indices)} samples)")
    
    return train_dataset, val_dataset, test_dataset
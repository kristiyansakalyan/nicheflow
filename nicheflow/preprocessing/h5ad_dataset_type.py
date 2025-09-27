import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class H5ADDatasetDataclass:
    # === Data === #
    X_pca: np.ndarray
    coords: np.ndarray
    ct: np.ndarray
    PCs: np.ndarray

    # === Timepoint info === #
    timepoints_ordered: list[str]
    timepoint_column: str
    timepoint_to_int: dict[str, int]
    timepoint_indices: dict[str, np.ndarray]

    # === Cell type info === #
    ct_column: str
    ct_ordered: list[str]
    ct_to_int: dict[str, int]

    # === Radius-Graph-Related === #
    timepoint_neighboring_indices: dict[str, dict[int, list[int]]]
    timepoint_num_neighbors: dict[str, int]
    subsampled_timepoint_idx: dict[str, list[int]]

    # === Parameters === #
    standardize_coordinates: bool
    radius: float
    dx: float
    dy: float

    # === Statistics == #
    stats: dict[str, dict[str, dict[str, np.ndarray]] | dict[str, np.ndarray]]

    # === Test-related == #
    test_microenvs: int


def load_h5ad_dataset_dataclass(filepath: str) -> H5ADDatasetDataclass:
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"The file does not exist: {fp}")

    with fp.open("rb") as file:
        ds: H5ADDatasetDataclass = pickle.load(file)

    return ds

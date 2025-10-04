import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from nicheflow.preprocessing.h5ad_dataset_type import H5ADDatasetDataclass


def create_kmeans_regions(
    ds: H5ADDatasetDataclass, timepoint_pc: dict[str, Data], k_regions: int, seed: int
) -> dict[str, dict[int, list[int]]]:
    timepoint_regions_to_idx: dict[str, dict[int, list[int]]] = {}

    for timepoint in ds.timepoints_ordered:
        coords = timepoint_pc[timepoint].pos.numpy()
        # Cluster the shape in regions
        kmeans = KMeans(n_clusters=k_regions, random_state=seed)
        region_labels = kmeans.fit_predict(coords)
        # Save a map from each region to the indices
        region_to_idxs = {i: np.argwhere(region_labels == i).squeeze() for i in range(k_regions)}
        timepoint_regions_to_idx[timepoint] = region_to_idxs

    return timepoint_regions_to_idx


def init_worker_rng(seed: int) -> np.random.Generator:
    """
    Initializes the worker-specific random number generator (RNG).

    When using `IterableDataset` with a `DataLoader` and `num_workers > 0`, each worker
    gets its own copy of the dataset. If all workers share the same RNG seed, they will
    produce identical data samples. This method ensures that each worker uses a distinct,
    deterministic seed (based on the base seed and worker ID), avoiding duplicate data.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        # Single-process (no workers): use base seed
        rng = np.random.default_rng(seed)
    else:
        # Worker-specific seed
        worker_seed = seed + info.id
        rng = np.random.default_rng(worker_seed)

    return rng

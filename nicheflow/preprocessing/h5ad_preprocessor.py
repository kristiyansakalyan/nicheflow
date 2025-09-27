import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scanpy import AnnData
from tqdm import tqdm

from nicheflow.preprocessing import H5ADDatasetDataclass
from nicheflow.utils import chunked_cdist_sum_argsort, grid_based_sampling_by_y, setup_logging

MIN_COVERAGE = 5
setup_logging()
_logger = logging.getLogger(__name__)


class H5ADPreprocessor:
    def __init__(
        self,
        timepoint_column: str,
        cell_type_column: str,
        timepoints_ordered: list[Any],
        standardize_coordinates: bool = True,
        radius: float = 0.15,
        dx: float = 0.15,
        dy: float = 0.2,
        device: str = "cpu",
        chunk_size: int = 1000,
    ) -> None:
        self.timepoint_column = timepoint_column
        self.ct_column = cell_type_column
        self.timepoints_ordered = timepoints_ordered
        self.standardize_coordinates = standardize_coordinates
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.chunk_size = chunk_size
        self.device = torch.device(device)

        # These will be set by the preprocessing
        self.timepoints = None
        self.timepoints_to_int = None

        # Cell types
        self.ct_ordered = None
        self.ct_to_int = None

        # Data
        self.ct = None
        self.X_pca = None
        self.coords = None
        self.PCs = None

        # Radius graph related
        self.subsampled_timepoint_idx: dict[str | float, list[int]] = {}
        self.timepoint_neighboring_indices: dict[float, dict[int, list[int]]] = {
            # timepoint: {
            #   index: list[indices]
            # }
        }
        self.timepoint_num_neighbors: dict[float, int] = {}

        # Test microenvironments per slice
        self.test_microenvs = None

        # Statistics that are needed to reverse the normalizations
        self.stats = {"coords": {}, "X_pca": {}}

    def preprocess_data(self, adata: AnnData) -> None:
        self.PCs = adata.varm["PCs"]
        self._prepare_timepoints_and_annotations(adata)
        self._normalize_coordinates_and_features(adata)
        self._compute_radius_graphs()
        self._subsample_centroids()

    def _prepare_timepoints_and_annotations(self, adata: AnnData) -> None:
        # Timepoints
        self.timepoint_to_int = {
            timepoint: i for i, timepoint in enumerate(self.timepoints_ordered)
        }
        self.timepoint_indices: dict[str, np.ndarray] = {
            t: np.where(adata.obs[self.timepoint_column] == t)[0] for t in self.timepoints_ordered
        }
        # Annotations
        self.ct_ordered = sorted(adata.obs[self.ct_column].cat.categories)
        self.ct_to_int = {annotation: i for i, annotation in enumerate(self.ct_ordered)}

    def _normalize_coordinates_and_features(self, adata: AnnData) -> None:
        method = "standardization" if self.standardize_coordinates else "min-max scaling"
        _logger.info(f"Preprocessing the spatial coordinates per timepoint with: {method}")

        self.coords = adata.obsm["spatial"]

        # Spatial normalization
        for timepoint in self.timepoints_ordered:
            indices = self.timepoint_indices[timepoint]
            if self.standardize_coordinates:
                coords_mean = self.coords[indices].mean(axis=0)
                coords_std = self.coords[indices].std(axis=0)
                # Save statistics
                self.stats["coords"][timepoint] = {"mean": coords_mean, "std": coords_std}

                self.coords[indices] = (self.coords[indices] - coords_mean) / coords_std
            else:
                coords_min = self.coords[indices].min(axis=0)
                coords_max = self.coords[indices].max(axis=0)
                # Save statistics
                self.stats["coords"][timepoint] = {"min": coords_min, "max": coords_max}
                self.coords[indices] = (self.coords[indices] - coords_min) / (
                    coords_max - coords_min
                )

        # Feature normalization
        X_pca_mean = adata.obsm["X_pca"].mean(axis=0)  # noqa: N806
        X_pca_std = adata.obsm["X_pca"].std(axis=0)  # noqa: N806

        # Save statistics
        self.stats["X_pca"] = {"mean": X_pca_mean, "std": X_pca_std}
        self.X_pca = (adata.obsm["X_pca"] - X_pca_mean) / X_pca_std

        # Save annotations for each cell.
        self.ct = np.array(adata.obs[self.ct_column])

    def _compute_radius_graphs(self) -> None:
        # Compute the radius graphs for each timepoint
        if self.device.type == "cpu":
            _logger.warning("Using `CPU`! Might be too slow!")
        elif self.device.type == "cuda":
            _logger.warning("Using `CUDA`! Reduce the `chunk_size` if OOM.")

        compute_iter = tqdm(self.timepoints_ordered)
        for timepoint in compute_iter:
            compute_iter.set_description(f"Computing radius graphs for timepoint: '{timepoint}'")

            # Get the indices and coords;
            indices = self.timepoint_indices[timepoint]
            coords_t = torch.Tensor(self.coords[indices]).to(self.device)

            # Fix the number of nodes per graph.
            num_neighbors, C_t_argsorted = chunked_cdist_sum_argsort(  # noqa: N806
                coords=coords_t, radius=self.radius, chunk_size=self.chunk_size
            )
            unique, counts = torch.unique(num_neighbors, return_counts=True)

            # Choose the most common N for the current radius.
            N = unique[counts.argmax()].cpu().numpy()  # noqa: N806
            self.timepoint_num_neighbors[timepoint] = N

            # Get the neighbor indices
            neighbor_indices = C_t_argsorted[:, :N].cpu().numpy()
            # Convert to dictionary for fast access
            neighbors_dict = {
                i: neighbor_indices for i, neighbor_indices in enumerate(neighbor_indices)
            }
            self.timepoint_neighboring_indices[timepoint] = neighbors_dict

            # Clear cuda memory
            del coords_t, num_neighbors, C_t_argsorted, unique, counts
            torch.cuda.empty_cache()

    def _subsample_centroids(self) -> None:
        if self.coords is None:
            raise ValueError("The coordinates must not be None at this point.")

        # Subsample the centroids.
        subsample_iter = tqdm(self.timepoints_ordered)
        for timepoint in subsample_iter:
            subsample_iter.set_description(
                f"Subsampling centroids t='{timepoint}' | dx={self.dx} | dy={self.dy}"
            )
            gt_indices = self.timepoint_indices[timepoint]
            gt = self.coords[gt_indices]

            pos_idx = grid_based_sampling_by_y(
                coords=gt,
                dx=self.dx,
                dy=self.dy,
            )
            self.subsampled_timepoint_idx[timepoint] = pos_idx

            # Validation
            subgraph_indices = np.unique(
                np.concatenate(
                    [self.timepoint_neighboring_indices[timepoint][idx] for idx in pos_idx]
                )
            )

            diff = np.abs(len(gt) - len(subgraph_indices))

            if diff > MIN_COVERAGE:
                _logger.warning(
                    "You should change the values for `dx` and `dy`."
                    + f"GT: {len(gt)} | Microenvironment cover: {len(subgraph_indices)}"
                )
        # Fix nodes by upsampling
        self.test_microenvs = max([len(x) for x in self.subsampled_timepoint_idx.values()])
        _logger.info(f"Fixing test microenvironments to {self.test_microenvs} per slice.")

        # Randomly sample additional indices without the ones already present.
        for timepoint in self.timepoints_ordered:
            length = len(self.timepoint_neighboring_indices[timepoint])
            subsampled_indices = self.subsampled_timepoint_idx[timepoint]
            n_upsample = max(self.test_microenvs - len(subsampled_indices), 0)

            if n_upsample != 0:
                choices = [i for i in range(length) if i not in subsampled_indices]
                choices = np.random.choice(choices, n_upsample, replace=False)
                self.subsampled_timepoint_idx[timepoint] = np.concatenate(
                    [self.subsampled_timepoint_idx[timepoint], choices]
                )

    def save(self, filepath: str) -> None:
        fp = Path(filepath)

        ds = H5ADDatasetDataclass(
            # Data
            X_pca=self.X_pca,
            coords=self.coords,
            ct=self.ct,
            PCs=self.PCs,
            # Timepoint info
            timepoints_ordered=self.timepoints_ordered,
            timepoint_column=self.timepoint_column,
            timepoint_to_int=self.timepoint_to_int,
            timepoint_indices=self.timepoint_indices,
            # Cell types
            ct_column=self.ct_column,
            ct_ordered=self.ct_ordered,
            ct_to_int=self.ct_to_int,
            # Radius-graph related
            timepoint_neighboring_indices=self.timepoint_neighboring_indices,
            timepoint_num_neighbors=self.timepoint_num_neighbors,
            subsampled_timepoint_idx=self.subsampled_timepoint_idx,
            # Parameters
            standardize_coordinates=self.standardize_coordinates,
            radius=self.radius,
            dx=self.dx,
            dy=self.dy,
            # Stats
            stats=self.stats,
            # Test info
            test_microenvs=self.test_microenvs,
        )

        with fp.open("wb") as file:
            pickle.dump(ds, file, protocol=pickle.HIGHEST_PROTOCOL)

        _logger.info(f"Saved file at: {fp}")

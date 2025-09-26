import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree  # type: ignore


def grid_based_sampling_by_y(
    coords: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Discretize y in fixed steps of dy. For each y-bin, find the subset of points
    that lie within [y_i, y_i + dy). Determine min_x and max_x for that subset,
    then create an x-grid (with spacing dx) across that bin's range.
    For each (x, y_center), pick the nearest real point using a KD-tree.

    Parameters
    ----------
    coords : np.ndarray
        The point coordinates (N, 2)
    dx : float
        dx to be used to discretize over x
    dy : float
        dy to be used to discretize over y

    Returns
    -------
    np.ndarray
        Unique integer indices of 'points' that were chosen.
    """
    # Ensure points is an N x 2 array
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:  # noqa: PLR2004
        raise ValueError("Points should be a (N,2) array")

    # Build a KD-tree for fast nearest-neighbor queries (on full set)
    tree = cKDTree(coords)  # type: ignore

    # 1) Compute overall bounding box
    min_xy = coords.min(axis=0)  # [min_x, min_y]
    max_xy = coords.max(axis=0)  # [max_x, max_y]
    _, min_y = min_xy
    _, max_y = max_xy

    # 2) Discretize y into bins: y in [y_i, y_i + dy)
    y_bins = np.arange(min_y, max_y, dy)

    selected_indices = []

    for y_start in y_bins:
        y_end = y_start + dy
        # We can sample at the "center" of this bin in y
        y_center = (y_start + y_end) / 2.0

        # 3) Find all points in this bin
        in_bin_mask = (coords[:, 1] >= y_start) & (coords[:, 1] <= y_end)
        bin_points = coords[in_bin_mask]
        if len(bin_points) == 0:
            # No points in this y-range, skip
            continue

        # 4) Determine min_x and max_x among bin_points
        bx_min = bin_points[:, 0].min()
        bx_max = bin_points[:, 0].max()

        # 5) Create grid of x-centers for this y-bin
        x_centers = np.arange(bx_min, bx_max, dx)

        for x_center in x_centers:
            # Query the global KD-tree for nearest neighbor
            _, idx = tree.query([x_center, y_center])
            selected_indices.append(idx)  # type: ignore

    # 6) Remove duplicates
    selected_indices = np.unique(selected_indices)  # type: ignore
    return selected_indices  # type: ignore


def chunked_cdist_sum_argsort(
    coords: torch.Tensor,
    radius: float,
    chunk_size: int = 1000,
    max_columns: int = 10_000,
):
    n_points = coords.shape[0]
    max_column = min(n_points, max_columns)

    n_neighbours = torch.zeros((n_points), dtype=torch.int32).to(coords.device)
    dist_argsorted = torch.zeros((n_points, max_column), dtype=torch.int32).to(coords.device)
    for i in range(0, n_points, chunk_size):
        i_end = min(i + chunk_size, n_points)

        cdist_matrix = torch.cdist(coords[i:i_end], coords)
        n_neighbours[i:i_end] = (cdist_matrix < radius).sum(dim=-1)
        dist_argsorted[i:i_end] = cdist_matrix.argsort(dim=-1)[:, :max_column]

        if cdist_matrix.device.type == "cuda":
            del cdist_matrix
            torch.cuda.empty_cache()

    return n_neighbours, dist_argsorted

import math
from abc import abstractmethod
from collections.abc import Generator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets.st_dataset_base import STDatasetBase, STTrainDataItem, STValDataItem
from nicheflow.preprocessing import load_h5ad_dataset_dataclass
from nicheflow.preprocessing.h5ad_dataset_type import H5ADDatasetDataclass
from nicheflow.utils.datasets import create_kmeans_regions, init_worker_rng
from nicheflow.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class RandomPointCloudDatasetBase(STDatasetBase):
    """
    Samples point clouds of size `N` uniformly from predefined spatial regions
    at each timepoint, then applies mini-batch optimal transport to align source
    and target points.

    The same dataset can be used to train both single-point and point cloud flow models.
    The difference lies in how the model consumes the data - either as entire point clouds
    or by individual points.
    """

    def __init__(
        self,
        ds: H5ADDatasetDataclass,
        seed: int = 2025,
        k_regions: int = 64,
        size_per_slice: int = 1024,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
    ) -> None:
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
        )
        self.size_per_slice = size_per_slice
        self.seed = seed
        self.k_regions = k_regions
        # Use a seeded generator for sampling the pairs
        # and sampling the microenvironments within the K regions
        self.rng: np.random.Generator = np.random.default_rng(self.seed)

        # Create KMeans regions
        self.timepoint_regions_to_idx: dict[str, dict[int, list[int]]] = create_kmeans_regions(
            ds=ds, timepoint_pc=self.timepoint_pc, k_regions=self.k_regions, seed=self.seed
        )

        # Precompute the number of cells we will sample per region
        self.size_per_region = self.size_per_slice // self.k_regions
        if self.size_per_region == 0:
            _logger.warning(
                "The point cloud size per slice must be larger than the number of regions!"
                + f"Got {self.size_per_slice} microenvironments but only {self.k_regions} regions."
            )
            _logger.warning("Setting the point cloud size per slice to 1")
            self.size_per_region = 1
            self.size_per_slice = k_regions

    @abstractmethod
    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        raise NotImplementedError(
            "The method `_get_timepoints` must be implemented in child classes!"
        )

    def _sample_cell_idxs(self, timepoint: str) -> list[int]:
        # Extract the region to centroids map for a slice at time t
        region_to_idxs = self.timepoint_regions_to_idx[timepoint]

        selected_idxs: list[int] = []
        for region_id, region_idxs_list in region_to_idxs.items():
            region_idxs = np.array(region_idxs_list)

            if len(region_idxs) < self.size_per_region:
                _logger.warning(
                    f"Region {region_id} at time {timepoint} has less cells than the size per region. "
                    + f"It has {len(region_idxs)} but we sample {self.size_per_region}. "
                    + "Using `replace=True` during sampling"
                )
                sampled = self.rng.choice(region_idxs, size=self.size_per_region, replace=True)
            else:
                sampled = self.rng.choice(region_idxs, size=self.size_per_region, replace=False)
            selected_idxs.extend(sampled)

        return selected_idxs

    def _ot(self, rpc_t1: Data, rpc_t2: Data) -> tuple[np.ndarray, np.ndarray]:
        features_size = rpc_t1.x.size(-1)
        coordinates_size = rpc_t1.pos.size(-1)

        source = torch.cat([rpc_t1.x, rpc_t1.pos], dim=-1)
        target = torch.cat([rpc_t2.x, rpc_t2.pos], dim=-1)

        # Apply the weighting
        lambda_tensor = torch.cat(
            [
                torch.repeat_interleave(self.ot_lambda, features_size),
                torch.repeat_interleave(1 - self.ot_lambda, coordinates_size),
            ]
        )
        source *= lambda_tensor
        target *= lambda_tensor

        # Now perform the OT
        pi = self.ot_plan_sampler.get_map(x0=source, x1=target)
        source_idxs, target_idxs = self.ot_plan_sampler.sample_map(
            pi, batch_size=self.size_per_slice, replace=False
        )

        return source_idxs, target_idxs

    def _get_rpcs_t1_t2(self, index: int | None) -> tuple[Data, Data, str, str]:
        t1, t2 = self._get_timepoints(index=index)

        # Sample a random point cloud uniformly over the regions
        rpc_idxs_t1 = self._sample_cell_idxs(t1)
        rpc_idxs_t2 = self._sample_cell_idxs(t2)

        rpc_t1 = self.timepoint_pc[t1].subgraph(torch.Tensor(rpc_idxs_t1).to(torch.int32))
        rpc_t2 = self.timepoint_pc[t2].subgraph(torch.Tensor(rpc_idxs_t2).to(torch.int32))

        source_idxs, target_idxs = self._ot(rpc_t1=rpc_t1, rpc_t2=rpc_t2)

        rpc_t1_resampled = Data(
            x=rpc_t1.x[source_idxs],
            pos=rpc_t1.pos[source_idxs],
            ct=rpc_t1.ct[source_idxs],
            t_ohe=rpc_t1.t_ohe,
        )

        rpc_t2_resampled = Data(
            x=rpc_t2.x[target_idxs],
            pos=rpc_t2.pos[target_idxs],
            ct=rpc_t2.ct[target_idxs],
            t_ohe=rpc_t2.t_ohe,
        )

        return rpc_t1_resampled, rpc_t2_resampled, t1, t2


class InfiniteRPCDataset(IterableDataset, RandomPointCloudDatasetBase):
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        size_per_slice: int = 1024,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
    ) -> None:
        ds = load_h5ad_dataset_dataclass(data_fp)
        super().__init__(
            ds=ds,
            seed=seed,
            k_regions=k_regions,
            size_per_slice=size_per_slice,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
        )

    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        # Every time we randomly select two consecutive slices
        pair_idx = self.rng.integers(self.num_pairs)
        t1, t2 = self.consecutive_pairs[pair_idx]
        return t1, t2

    def __iter__(self) -> Generator[STTrainDataItem]:
        self.rng = init_worker_rng(seed=self.seed)
        while True:
            # We use index = None because in the infinite training dataset, we are
            # doing random sampling
            rpc_t1, rpc_t2, t1, t2 = self._get_rpcs_t1_t2(index=None)
            yield {
                # First random point cloud
                "X_t1": rpc_t1.x,
                "pos_t1": rpc_t1.pos,
                "t1_ohe": self.timepoint_pc[t1].t_ohe,
                # Second random point cloud
                "X_t2": rpc_t2.x,
                "pos_t2": rpc_t2.pos,
                "t2_ohe": self.timepoint_pc[t2].t_ohe,
            }


class TestRPCDataset(RandomPointCloudDatasetBase, Dataset):
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        size_per_slice: int = 1024,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        upsample_factor: int = 1,
    ) -> None:
        ds = load_h5ad_dataset_dataclass(data_fp)
        super().__init__(
            ds=ds,
            seed=seed,
            k_regions=k_regions,
            size_per_slice=size_per_slice,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
        )
        self.upsample_factor = upsample_factor

        # Create test pairs needed to reach the desired upsampling factor
        self.test_pairs: list[tuple[str, str]] = []
        for t1, t2 in self.consecutive_pairs:
            num_cells_t2 = self.timepoint_pc[t2].num_nodes
            num_samples = math.ceil(num_cells_t2 / self.size_per_slice)
            self.test_pairs.extend([(t1, t2)] * num_samples * self.upsample_factor)

    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        if index is None:
            raise ValueError("Index in TestRPCDataset must not be None!")

        if index >= len(self):
            raise ValueError(f"Index `{index}` is out of bounds!")

        return self.test_pairs[index]

    def __getitem__(self, index: int) -> STValDataItem:
        rpc_t1, rpc_t2, t1, t2 = self._get_rpcs_t1_t2(index=index)
        return {
            # First random point cloud
            "X_t1": rpc_t1.x,
            "pos_t1": rpc_t1.pos,
            "t1_ohe": self.timepoint_pc[t1].t_ohe,
            # Second random point cloud
            "X_t2": rpc_t2.x,
            "pos_t2": rpc_t2.pos,
            "t2_ohe": self.timepoint_pc[t2].t_ohe,
            # We need to perform global evaluation, therefore we need
            # all the positions and all the cell type annotations
            "global_pos_t2": self.timepoint_pc[t2].pos,
            "global_ct_t2": self.timepoint_pc[t2].ct,
        }

    def __len__(self) -> int:
        return len(self.test_pairs)

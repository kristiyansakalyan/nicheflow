from abc import abstractmethod
from collections.abc import Generator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets.st_dataset_base import STDatasetBase, STTrainDataItem, STValDataItem
from nicheflow.preprocessing import H5ADDatasetDataclass, load_h5ad_dataset_dataclass
from nicheflow.utils.datasets import create_kmeans_regions, init_worker_rng
from nicheflow.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class MicroEnvDatasetBase(STDatasetBase):
    def __init__(
        self,
        ds: H5ADDatasetDataclass,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
    ) -> None:
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
        )

        # Must be set in child classes
        self.n_microenvs_per_slice: int
        self.per_microenv_transforms = per_microenv_transforms
        # Save the important attributes from the dataset class
        self.timepoint_neighboring_indices = ds.timepoint_neighboring_indices

    @abstractmethod
    def _sample_microenvs_idxs(self, timepoint: str) -> list[list[int]]:
        raise NotImplementedError(
            "The method `_sample_microenvs_idxs` must be implemented in child classes!"
        )

    @abstractmethod
    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        raise NotImplementedError(
            "The method `_get_timepoints` must be implemented in child classes!"
        )

    def _mini_batch_ot(
        self, microenvs_t1: list[Data], microenvs_t2: list[Data]
    ) -> tuple[list[Data], list[Data]]:
        X_t1 = torch.stack([el.x for el in microenvs_t1])
        X_t2 = torch.stack([el.x for el in microenvs_t2])
        pos_t1 = torch.stack([el.pos for el in microenvs_t1])
        pos_t2 = torch.stack([el.pos for el in microenvs_t2])

        features_size = X_t1.size(-1)
        coordinates_size = pos_t1.size(-1)

        # Concatenated the gene expresion profiles with the positions and
        # pool by computing the mean gene expression profile
        # and the centroid of each microenvironment
        # The dimensions are (n_microenvs_per_slice, n_cells, genes + coordinates)
        source = torch.cat([X_t1, pos_t1], dim=-1).mean(dim=1)
        target = torch.cat([X_t2, pos_t2], dim=-1).mean(dim=1)

        # Apply the weighting on the pooled microenvironment representations
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
            pi, batch_size=self.n_microenvs_per_slice, replace=False
        )

        resampled_microenvs_t1 = [microenvs_t1[i] for i in source_idxs]
        resampled_microenvs_t2 = [microenvs_t2[i] for i in target_idxs]

        return resampled_microenvs_t1, resampled_microenvs_t2

    def _get_microenvs_t1_t2(self, index: int | None) -> tuple[list[Data], list[Data], str, str]:
        t1, t2 = self._get_timepoints(index=index)

        # Sample microenvironment centroids uniformly over the defined spatial regions
        microenv_idxs_t1 = self._sample_microenvs_idxs(t1)
        microenv_idxs_t2 = self._sample_microenvs_idxs(t2)

        # Create the torch geometric data objects
        microenvs_t1: list[Data] = [
            self.per_microenv_transforms(
                self.timepoint_pc[t1].subgraph(torch.Tensor(idx).to(torch.int32))
            )
            for idx in microenv_idxs_t1
        ]
        microenvs_t2: list[Data] = [
            self.per_microenv_transforms(
                self.timepoint_pc[t2].subgraph(torch.Tensor(idx).to(torch.int32))
            )
            for idx in microenv_idxs_t2
        ]

        # Perform mini-batch OT
        microenvs_t1, microenvs_t2 = self._mini_batch_ot(
            microenvs_t1=microenvs_t1, microenvs_t2=microenvs_t2
        )

        return microenvs_t1, microenvs_t2, t1, t2


class InfiniteMicroEnvDataset(IterableDataset, MicroEnvDatasetBase):
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        n_microenvs_per_slice: int = 256,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
    ) -> None:
        ds = load_h5ad_dataset_dataclass(data_fp)
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
            per_microenv_transforms=per_microenv_transforms,
        )
        self.seed = seed
        # Use a seeded generator for sampling the pairs
        # and sampling the microenvironments within the K regions
        self.rng = None

        self.k_regions = k_regions
        self.n_microenvs_per_slice = n_microenvs_per_slice

        # Create KMeans regions
        self.timepoint_regions_to_idx: dict[str, dict[int, list[int]]] = create_kmeans_regions(
            ds=ds, timepoint_pc=self.timepoint_pc, k_regions=self.k_regions, seed=self.seed
        )

        # Precompute the number of microenvironments we will sample per region
        self.n_microenvs_per_region = self.n_microenvs_per_slice // self.k_regions
        if self.n_microenvs_per_region == 0:
            _logger.warning(
                "The number of microenvironments per slice must be larger than the number of regions!"
                + f"Got {self.n_microenvs_per_slice} microenvironments but only {self.k_regions} regions."
            )
            _logger.warning("Setting the microenvironments per slice to 1")
            self.n_microenvs_per_slice = k_regions
            self.n_microenvs_per_region = 1

    def _sample_microenvs_idxs(self, timepoint: str) -> list[list[int]]:
        # Extract the region to centroids map for a slice at time t
        region_to_idxs = self.timepoint_regions_to_idx[timepoint]

        selected_idxs: list[int] = []
        for region_id, region_idxs_list in region_to_idxs.items():
            region_idxs = np.array(region_idxs_list)

            if len(region_idxs) < self.n_microenvs_per_region:
                _logger.warning(
                    f"Region {region_id} at time {timepoint} has less microenvironemnts "
                    + f"than the microenviornments per region. It has {len(region_idxs)} "
                    + f"but we sample {self.n_microenvs_per_region}. Using `replace=True`"
                    + " during sampling."
                )
                sampled = self.rng.choice(
                    region_idxs, size=self.n_microenvs_per_region, replace=True
                )
            else:
                sampled = self.rng.choice(
                    region_idxs, size=self.n_microenvs_per_region, replace=False
                )
            selected_idxs.extend(sampled)

        return [self.timepoint_neighboring_indices[timepoint][i] for i in selected_idxs]

    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        # Every time we randomly select two consecutive slices
        pair_idx = self.rng.integers(self.num_pairs)
        t1, t2 = self.consecutive_pairs[pair_idx]
        return t1, t2

    def __iter__(self) -> Generator[STTrainDataItem]:
        self.rng = init_worker_rng(seed=self.seed)
        while True:
            # We use index = None because in the infinite training dataset, we are
            # doing random sampling.
            microenvs_t1, microenvs_t2, t1, t2 = self._get_microenvs_t1_t2(index=None)
            yield {
                # First slice
                "X_t1": torch.stack([pc.x for pc in microenvs_t1]),
                "pos_t1": torch.stack([pc.pos for pc in microenvs_t1]),
                "t1_ohe": self.timepoint_pc[t1].t_ohe,
                # Second slice
                "X_t2": torch.stack([pc.x for pc in microenvs_t2]),
                "pos_t2": torch.stack([pc.pos for pc in microenvs_t2]),
                "t2_ohe": self.timepoint_pc[t2].t_ohe,
            }


class TestMicroEnvDataset(Dataset, MicroEnvDatasetBase):
    def __init__(
        self,
        data_fp: str,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
        upsample_factor: int = 1,
    ) -> None:
        ds = load_h5ad_dataset_dataclass(data_fp)
        super().__init__(
            ds=ds,
            ot_plan_sampler=ot_plan_sampler,
            ot_lambda=ot_lambda,
            per_pc_transforms=per_pc_transforms,
            per_microenv_transforms=per_microenv_transforms,
        )

        # We sample each consecutive pair `upsample_factor` times
        self.upsample_factor = upsample_factor
        self.length = upsample_factor * self.num_pairs

        self.n_microenvs_per_slice = ds.test_microenvs
        self.subsampled_timepoint_idx = ds.subsampled_timepoint_idx

    def _sample_microenvs_idxs(self, timepoint: str) -> list[list[int]]:
        # We are always using the grid-like subsampled centroids for testing
        return [
            self.timepoint_neighboring_indices[timepoint][i]
            for i in self.subsampled_timepoint_idx[timepoint]
        ]

    def _get_timepoints(self, index: int | None) -> tuple[str, str]:
        if index is None:
            raise ValueError("Index in the TestMicroEnvDataset must not be None.")

        pair_idx = index // self.upsample_factor

        if pair_idx >= len(self.consecutive_pairs):
            raise ValueError(f"Index `{index}` is out of bounds!")

        t1, t2 = self.consecutive_pairs[pair_idx]
        return t1, t2

    def __getitem__(self, index: int) -> STValDataItem:
        microenvs_t1, microenvs_t2, t1, t2 = self._get_microenvs_t1_t2(index=index)
        return {
            # First slice
            "X_t1": torch.stack([pc.x for pc in microenvs_t1]),
            "pos_t1": torch.stack([pc.pos for pc in microenvs_t1]),
            "t1_ohe": self.timepoint_pc[t1].t_ohe,
            # Second slice
            "X_t2": torch.stack([pc.x for pc in microenvs_t2]),
            "pos_t2": torch.stack([pc.pos for pc in microenvs_t2]),
            "t2_ohe": self.timepoint_pc[t2].t_ohe,
            # We need to perform global evaluation, therefore we need
            # all the positions and all the cell type annotations
            "global_pos_t2": self.timepoint_pc[t2].pos,
            "global_ct_t2": self.timepoint_pc[t2].ct,
        }

    def __len__(self) -> int:
        return self.length

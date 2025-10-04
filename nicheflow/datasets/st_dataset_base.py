from abc import ABC
from typing import TypedDict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.preprocessing import H5ADDatasetDataclass
from nicheflow.transforms import OHESlide
from nicheflow.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class STTrainDataItem(TypedDict):
    X_t1: torch.Tensor
    pos_t1: torch.Tensor
    t1_ohe: torch.Tensor

    X_t2: torch.Tensor
    pos_t2: torch.Tensor
    t2_ohe: torch.Tensor


class STValDataItem(TypedDict, STTrainDataItem):
    global_pos_t2: torch.Tensor
    global_ct_t2: torch.Tensor


class STTrainDataBatch(TypedDict, STTrainDataItem):
    mask_t1: torch.Tensor
    mask_t2: torch.Tensor


class STDatasetBase(ABC):
    def __init__(
        self,
        ds: H5ADDatasetDataclass,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        ot_lambda: float = 0.1,
        per_pc_transforms: Compose = Compose([]),
    ) -> None:
        super().__init__()
        self.ot_plan_sampler = ot_plan_sampler
        self.ot_lambda = torch.tensor(ot_lambda)
        # Make sure that we always one hot encode the timestep
        self.per_pc_transforms = Compose(
            [*per_pc_transforms.transforms, OHESlide(size=len(ds.timepoints_ordered))]
        )

        # Create per timepoint global point clouds
        self.timepoint_pc: dict[str, Data] = {}
        self._compute_timepoint_pc(ds=ds)

        # Create (t_i, t_{i+1}) pairs
        self.consecutive_pairs: list[tuple[str, str]] = list(
            zip(ds.timepoints_ordered[:-1], ds.timepoints_ordered[1:], strict=False)
        )
        self.num_pairs = len(self.consecutive_pairs)

    def _compute_timepoint_pc(self, ds: H5ADDatasetDataclass) -> None:
        _logger.info("Creating per timepoint PyTorch Geoemtric Data objects")

        ct_get_vec = np.vectorize(ds.ct_to_int.get)

        for timepoint in ds.timepoints_ordered:
            indices = ds.timepoint_indices[timepoint]
            self.timepoint_pc[timepoint] = self.per_pc_transforms(
                Data(
                    x=torch.Tensor(ds.X_pca[indices]),
                    pos=torch.Tensor(ds.coords[indices]),
                    ct=torch.Tensor(ct_get_vec(ds.ct[indices])),
                    t_ohe=torch.Tensor([ds.timepoint_to_int[timepoint]]),
                )
            )

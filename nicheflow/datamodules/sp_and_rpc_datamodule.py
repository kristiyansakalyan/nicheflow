from abc import ABC, abstractmethod
from collections.abc import Callable

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets import InfiniteRPCDataset, STTrainDataItem, STValDataItem, TestRPCDataset
from nicheflow.utils import (
    rpc_transformer_train_collate,
    rpc_transformer_val_collate,
    single_point_train_collate,
    single_point_val_collate,
)


class BaseRPCDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        size_per_slice: int = 1024,
        ot_lambda: float = 0.1,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        per_pc_transforms: Compose = Compose([]),
        val_upsample_factor: int = 1,
        train_batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.val_upsample_factor = val_upsample_factor
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        self.common_dataset_args = {
            "data_fp": data_fp,
            "seed": seed,
            "k_regions": k_regions,
            "size_per_slice": size_per_slice,
            "ot_plan_sampler": ot_plan_sampler,
            "ot_lambda": ot_lambda,
            "per_pc_transforms": per_pc_transforms,
        }

    @property
    @abstractmethod
    def collate_fn_train(self) -> Callable[[list[STTrainDataItem]], STTrainDataItem]:
        raise NotImplementedError("The collate_fn_train property must be set in child classes!")

    @property
    @abstractmethod
    def collate_fn_val(self) -> Callable[[list[STValDataItem]], STValDataItem]:
        raise NotImplementedError("The collate_fn_val property must be set in child classes!")

    def prepare_data(self) -> None:
        self.train_dataset = InfiniteRPCDataset(**self.common_dataset_args)
        self.test_dataset = TestRPCDataset(
            **self.common_dataset_args, upsample_factor=self.val_upsample_factor
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_train,
            # TODO: Why is this an issue?
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn_val,
            # TODO: Why is this an issue?
            # pin_memory=True,
        )


class RPCDataModule(BaseRPCDataModule):
    @property
    def collate_fn_train(self) -> Callable[[list[STTrainDataItem]], STTrainDataItem]:
        return rpc_transformer_train_collate

    @property
    def collate_fn_val(self) -> Callable[[list[STValDataItem]], STValDataItem]:
        return rpc_transformer_val_collate


class SinglePointDataModule(BaseRPCDataModule):
    @property
    def collate_fn_train(self) -> Callable[[list[STTrainDataItem]], STTrainDataItem]:
        return single_point_train_collate

    @property
    def collate_fn_val(self) -> Callable[[list[STValDataItem]], STValDataItem]:
        return single_point_val_collate


__all__ = ["RPCDataModule", "SinglePointDataModule"]

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets import InfiniteRPCDataset, TestRPCDataset
from nicheflow.utils import sp_rpc_train_collate, sp_rpc_val_collate


class BaseRPCDataModule(LightningDataModule):
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
            collate_fn=sp_rpc_train_collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=sp_rpc_val_collate,
            pin_memory=True,
        )


class RPCDataModule(BaseRPCDataModule):
    pass


class SinglePointDataModule(BaseRPCDataModule):
    pass


__all__ = ["RPCDataModule", "SinglePointDataModule"]

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from torchcfm import OTPlanSampler

from nicheflow.datasets import InfiniteMicroEnvDataset, TestMicroEnvDataset
from nicheflow.utils import microenv_train_collate, microenv_val_collate


class MicroEnvDataModule(LightningDataModule):
    def __init__(
        self,
        data_fp: str,
        seed: int = 2025,
        k_regions: int = 64,
        n_microenvs_per_slice: int = 256,
        ot_lambda: float = 0.1,
        ot_plan_sampler: OTPlanSampler = OTPlanSampler(method="exact"),
        per_pc_transforms: Compose = Compose([]),
        per_microenv_transforms: Compose = Compose([]),
        val_upsample_factor: int = 1,
        train_batch_size: int = 16,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.k_regions = k_regions
        self.n_microenvs_per_slice = n_microenvs_per_slice
        self.val_upsample_factor = val_upsample_factor

        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        self.common_dataset_args = {
            "data_fp": data_fp,
            "ot_lambda": ot_lambda,
            "ot_plan_sampler": ot_plan_sampler,
            "per_pc_transforms": per_pc_transforms,
            "per_microenv_transforms": per_microenv_transforms,
        }

    def prepare_data(self) -> None:
        self.train_dataset = InfiniteMicroEnvDataset(
            **self.common_dataset_args,
            seed=self.seed,
            k_regions=self.k_regions,
            n_microenvs_per_slice=self.n_microenvs_per_slice,
        )

        self.test_dataset = TestMicroEnvDataset(
            **self.common_dataset_args, upsample_factor=self.val_upsample_factor
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=microenv_train_collate,
        )

    def eval_dl(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=microenv_val_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.eval_dl()

    def test_dataloader(self) -> DataLoader:
        return self.eval_dl()

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from nicheflow.datasets import H5ADCTDataset


class H5ADCTDataModule(LightningDataModule):
    def __init__(
        self,
        data_fp: str,
        split_seed: int = 2025,
        train_batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_fp = data_fp
        self.split_seed = int(split_seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def prepare_data(self) -> None:
        self.dataset = H5ADCTDataset(self.data_fp)

    def setup(self, stage: str) -> None:
        dataset_size = len(self.dataset)
        train_size = int(self.train_ratio * dataset_size)
        val_size = int(self.val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Use split_seed to ensure deterministic splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

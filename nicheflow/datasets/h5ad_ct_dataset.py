from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from nicheflow.preprocessing.h5ad_dataset_type import load_h5ad_dataset_dataclass


class CellTypeBatch(TypedDict):
    X: torch.Tensor
    y: torch.Tensor


class H5ADCTDataset(Dataset):
    def __init__(self, filepath: str) -> None:
        ds = load_h5ad_dataset_dataclass(filepath=filepath)

        # PCA-reduced gene expressions
        self.X = torch.Tensor(ds.X_pca)

        # The cell types
        ct_to_int_vec = np.vectorize(ds.ct_to_int.get)
        self.ct = torch.Tensor(ct_to_int_vec(ds.ct)).to(torch.long)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, index: int) -> CellTypeBatch:
        if index > len(self) or index < 0:
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")

        return {"X": self.X[index], "y": self.ct[index]}

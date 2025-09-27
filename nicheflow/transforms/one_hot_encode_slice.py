import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("ohe_slide")
class OHESlide(BaseTransform):
    """One hot encode the slide"""

    def __init__(self: "OHESlide", size: int = 3) -> None:
        super().__init__()
        self.size = size

    def forward(self: "OHESlide", data: Data) -> Data:
        t_ohe = torch.zeros(self.size)
        if data.t_ohe.size(0) != 1:
            raise ValueError("Data attribute `t_ohe` is not of shape (1)")
        t_ohe[data.t_ohe.to(torch.int)] = 1
        data.t_ohe = t_ohe
        return data

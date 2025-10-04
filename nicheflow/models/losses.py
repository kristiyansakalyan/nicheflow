from abc import ABC, abstractmethod
from typing import TypedDict

import torch
import torch.nn.functional as torchfunc
from torch import Tensor, nn


class FlowLosses(TypedDict):
    loss: Tensor
    loss_x: Tensor
    loss_pos: Tensor


class FlowLoss(nn.Module, ABC):
    def __init__(self, lambda_features: float, lambda_pos: float) -> None:
        super().__init__()
        self.lambda_features = lambda_features
        self.lambda_pos = lambda_pos

    @abstractmethod
    def forward(self, gt_x: Tensor, gt_pos: Tensor, pred_x: Tensor, pred_pos: Tensor) -> FlowLosses:
        raise NotImplementedError("The `forward` method of the loss has to be implemented!")


class BaseMSEFlowLoss(FlowLoss):
    """
    Base class for MSE-based flow losses over gene expressions (X) and positions (pos).

    Used by both:
    - Conditional Flow Matching (CFM)
    - Gaussian Variational Flow Matching (VFM)

    This class assumes the targets have already been computed correctly outside.
    """

    def forward(self, gt_x: Tensor, gt_pos: Tensor, pred_x: Tensor, pred_pos: Tensor) -> FlowLosses:
        loss_x = torchfunc.mse_loss(pred_x, gt_x) * self.lambda_features
        loss_pos = torchfunc.mse_loss(pred_pos, gt_pos) * self.lambda_pos
        loss = loss_x + loss_pos

        return {"loss": loss, "loss_x": loss_x, "loss_pos": loss_pos}


class CFMLoss(BaseMSEFlowLoss):
    """
    Loss for Conditional Flow Matching (CFM).

    Regresses the conditional vector field (the derivative of the interpolation).
    """

    pass


class GVFMLoss(BaseMSEFlowLoss):
    """
    Loss for Gaussian Variational Flow Matching (VFM).

    Regresses the final states directly.
    """

    pass


class GLVFMLoss(FlowLoss):
    def forward(self, gt_x: Tensor, gt_pos: Tensor, pred_x: Tensor, pred_pos: Tensor) -> FlowLosses:
        loss_x = torchfunc.mse_loss(input=pred_x, target=gt_x) * self.lambda_features
        loss_pos = torch.mean(torch.abs(pred_pos - gt_pos)) * self.lambda_pos
        loss = loss_x + loss_pos

        return {"loss": loss, "loss_x": loss_x, "loss_pos": loss_pos}


__all__ = ["CFMLoss", "FlowLoss", "FlowLosses", "GLVFMLoss", "GVFMLoss"]

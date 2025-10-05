from abc import ABC, abstractmethod
from functools import partial
from typing import Generic, Literal, TypeVar

import torch
from torch import Tensor, nn
from torchdyn.core import NeuralODE

from nicheflow.datasets import STTrainDataBatch, STValDataItem
from nicheflow.models.backbones import PointCloudTransformer, SinglePointMLP
from nicheflow.models.losses import CFMLoss, FlowLoss, FlowLosses, GLVFMLoss, GVFMLoss

BackboneType = TypeVar("BackboneType", bound=nn.Module)
VFMObjective = Literal["GVFM", "GLVFM"]
EPS = 1e-8


class FlowVariant(ABC):
    def __init__(self, lambda_features: float, lambda_pos: float) -> None:
        super().__init__()
        self.lambda_features = lambda_features
        self.lambda_pos = lambda_pos

    @abstractmethod
    def get_train_target(self, x1: Tensor, x0: Tensor) -> Tensor:
        raise NotImplementedError(
            "The `get_train_target` method has to be implemented in the child classes!"
        )

    @abstractmethod
    def get_vf(
        self, x_t: Tensor, pos_t: Tensor, x_pred: Tensor, pos_pred: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("The `get_vf` method has to be implemented in the child classes!")

    @abstractmethod
    def get_objective(self) -> FlowLoss:
        raise NotImplementedError("The factory method `get_objective` has to be implemented!")


class CFM(FlowVariant):
    def get_train_target(self, x1: Tensor, x0: Tensor) -> Tensor:
        return x1 - x0

    def get_vf(
        self, x_t: Tensor, pos_t: Tensor, x_pred: Tensor, pos_pred: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        return x_pred, pos_pred

    def get_objective(self) -> FlowLoss:
        return CFMLoss(lambda_features=self.lambda_features, lambda_pos=self.lambda_pos)


class VFM(FlowVariant):
    def __init__(
        self, lambda_features: float, lambda_pos: float, vfm_objective: VFMObjective
    ) -> None:
        super().__init__(lambda_features=lambda_features, lambda_pos=lambda_pos)
        self.vfm_objective = vfm_objective

    def get_train_target(self, x1: Tensor, x0: Tensor) -> Tensor:
        return x1

    def get_vf(
        self, x_t: Tensor, pos_t: Tensor, x_pred: Tensor, pos_pred: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        t_unsq = t[:, None, None]

        x_vf = (x_pred - x_t) / (1 - t_unsq + EPS)
        pos_vf = (pos_pred - pos_t) / (1 - t_unsq + EPS)
        return x_vf, pos_vf

    def get_objective(self) -> FlowLoss:
        if self.vfm_objective == "GVFM":
            return GVFMLoss(lambda_features=self.lambda_features, lambda_pos=self.lambda_pos)
        if self.vfm_objective == "GLVFM":
            return GLVFMLoss(lambda_features=self.lambda_features, lambda_pos=self.lambda_pos)

        raise ValueError(
            f"The VFM objective can only be `GVFM` or `GLVFM` but got `{self.vfm_objective}`"
        )


class BaseFlow(nn.Module, Generic[BackboneType], ABC):
    def __init__(
        self,
        backbone: BackboneType,
        variant: FlowVariant,
        num_steps: int = 10,
        solver: str = "euler",
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.variant = variant
        self.objective = self.variant.get_objective()
        self.num_steps = num_steps
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    @abstractmethod
    def _backbone_forward(
        self,
        x_cond: Tensor,
        pos_cond: Tensor,
        ohe_cond: Tensor,
        x_target: Tensor,
        pos_target: Tensor,
        ohe_target: Tensor,
        t: Tensor,
        mask_cond: Tensor | None = None,
        mask_target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "The `backbone_forward` method has to be implemented in the child classes!"
        )

    def interpolate(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        return (t * x1) + ((1 - t) * x0)

    def loss(self, batch: STTrainDataBatch) -> FlowLosses:
        B = batch["X_t1"].size(0)
        t = torch.rand(B, device=batch["X_t1"].device)

        # Gaussian noise
        X_t2_0 = torch.randn_like(batch["X_t2"])
        pos_t2_0 = torch.randn_like(batch["pos_t2"])

        t_unsq = t[:, None, None]
        X_t2_t = self.interpolate(X_t2_0, batch["X_t2"], t_unsq)
        pos_t2_t = self.interpolate(pos_t2_0, batch["pos_t2"], t_unsq)

        # Predict
        pred_x, pred_pos = self._backbone_forward(
            x_cond=batch["X_t1"],
            pos_cond=batch["pos_t1"],
            ohe_cond=batch["t1_ohe"],
            x_target=X_t2_t,
            pos_target=pos_t2_t,
            ohe_target=batch["t2_ohe"],
            t=t,
            mask_cond=batch.get("mask_t1", None),
            mask_target=batch.get("mask_t2", None),
        )

        gt_x = self.variant.get_train_target(x1=batch["X_t2"], x0=X_t2_0)
        gt_pos = self.variant.get_train_target(x1=batch["pos_t2"], x0=pos_t2_0)

        return self.objective(gt_x=gt_x, gt_pos=gt_pos, pred_x=pred_x, pred_pos=pred_pos)

    @torch.no_grad()
    def sample(self, batch: STValDataItem) -> tuple[list[Tensor], list[Tensor]]:
        X_t2_0 = torch.randn_like(batch["X_t2"])
        pos_t2_0 = torch.randn_like(batch["pos_t2"])

        # Create initial state
        X_0 = torch.cat([X_t2_0, pos_t2_0], dim=-1)

        def vector_field(
            t: Tensor,
            X_t: Tensor,
            cond: tuple[Tensor, Tensor, Tensor, Tensor],
            args=None,
        ) -> Tensor:
            B = X_t.size(0)
            t_rep = t.repeat(B)

            X_t1, pos_t1, t1_ohe, t2_ohe = cond
            X_t2_t = X_t[..., : self.backbone.pca_dim]
            pos_t2_t = X_t[..., self.backbone.pca_dim :]

            pred_x, pred_pos = self._backbone_forward(
                x_cond=X_t1,
                pos_cond=pos_t1,
                ohe_cond=t1_ohe,
                x_target=X_t2_t,
                pos_target=pos_t2_t,
                ohe_target=t2_ohe,
                t=t_rep,
                # During validation we always use all microenvironments from a **single**
                # slice. Therefore, we do not use any sort of masking as this is only required
                # when we mix multiple slices and we have different number of cells within each
                # slice.
                mask_cond=None,
                mask_target=None,
            )

            vf_x, vf_pos = self.variant.get_vf(
                x_t=X_t2_t, pos_t=pos_t2_t, x_pred=pred_x, pos_pred=pred_pos, t=t_rep
            )

            return torch.cat([vf_x, vf_pos], dim=-1)

        vf_func = partial(
            vector_field, cond=(batch["X_t1"], batch["pos_t1"], batch["t1_ohe"], batch["t2_ohe"])
        )

        t_span = torch.linspace(0, 1, self.num_steps, device=X_0.device)

        node = NeuralODE(
            vf_func, solver=self.solver, sensitivity="adjoint", atol=self.atol, rtol=self.rtol
        )

        out_traj = node.trajectory(X_0, t_span)
        X_traj = [out[..., : self.backbone.pca_dim] for out in out_traj]
        pos_traj = [out[..., self.backbone.pca_dim :] for out in out_traj]

        return X_traj, pos_traj


class SinglePointFlow(BaseFlow[SinglePointMLP]):
    def _backbone_forward(
        self,
        x_cond: Tensor,
        pos_cond: Tensor,
        ohe_cond: Tensor,
        x_target: Tensor,
        pos_target: Tensor,
        ohe_target: Tensor,
        t: Tensor,
        mask_cond: Tensor | None = None,
        mask_target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.backbone(
            x_cond=x_cond,
            pos_cond=pos_cond,
            ohe_cond=ohe_cond,
            x_target=x_target,
            pos_target=pos_target,
            ohe_target=ohe_target,
            t=t,
        )


class PointCloudFlow(BaseFlow[PointCloudTransformer]):
    def _backbone_forward(
        self,
        x_cond: Tensor,
        pos_cond: Tensor,
        ohe_cond: Tensor,
        x_target: Tensor,
        pos_target: Tensor,
        ohe_target: Tensor,
        t: Tensor,
        mask_cond: Tensor | None = None,
        mask_target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.backbone(
            pcs_cond=x_cond,
            pos_cond=pos_cond,
            ohe_cond=ohe_cond,
            pcs_target=x_target,
            pos_target=pos_target,
            ohe_target=ohe_target,
            t_target=t,
            mask_condition=mask_cond,
            mask_target=mask_target,
        )

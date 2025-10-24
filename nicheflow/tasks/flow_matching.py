import math
from typing import Any, Literal

import torch
import torchmetrics as tm
from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nicheflow.datasets import STTrainDataBatch, STValDataItem
from nicheflow.models import CTClassifierNet, FlowLosses, PointCloudFlow, SinglePointFlow


def get_nn_class_metrics(num_classes: int) -> tm.MetricCollection:
    return tm.MetricCollection(
        {
            "f1": tm.F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
            "accuracy": tm.Accuracy(task="multiclass", num_classes=num_classes),
            "precision": tm.Precision(
                task="multiclass", num_classes=num_classes, average="weighted"
            ),
            "recall": tm.Recall(task="multiclass", num_classes=num_classes, average="weighted"),
        }
    )


def get_psd_metrics() -> tm.MetricCollection:
    return tm.MetricCollection(
        {
            "mean": tm.MeanMetric(),
            "max": tm.MaxMetric(),
        }
    )


def get_reg_metrics() -> tm.MetricCollection:
    return tm.MetricCollection({"mse": tm.MeanSquaredError(), "mae": tm.MeanAbsoluteError()})


def load_classifier(classifier: CTClassifierNet, ckpt_path: str) -> None:
    classifier_ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device("cpu"))
    classifier.load_state_dict(
        {
            key.replace("net.net.", "net."): value
            for key, value in classifier_ckpt["state_dict"].items()
        }
    )


def nn_of_x_in_y(
    x: torch.Tensor, y: torch.Tensor, chunk_size: int = 5000
) -> tuple[torch.Tensor, torch.Tensor]:
    idxs: list[torch.Tensor] = []
    dists: list[torch.Tensor] = []

    for index in range(math.ceil(x.size(0) / chunk_size)):
        start = index * chunk_size
        end = (index + 1) * chunk_size
        D = torch.cdist(x[start:end], y)

        min_dist, min_idx = torch.min(D, dim=-1)
        idxs.append(min_idx)
        dists.append(min_dist)

    return torch.cat(idxs), torch.cat(dists)


class ShapeToPointDistance:
    def __init__(
        self,
        n_slices: int,
        device: torch.device,
        prefix: Literal["val", "test"],
        chunk_size: int = 5000,
    ) -> None:
        self.n_slices = n_slices
        self.device = device
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.reset()

    def reset(self) -> None:
        self.preds_per_timepoint: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(self.n_slices)
        }
        self.gts_per_timepoint: dict[int, torch.Tensor | None] = {
            i: None for i in range(self.n_slices)
        }

    def update(self, pos_pred: torch.Tensor, pos_gt: torch.Tensor, timepoint: int) -> None:
        # Update the predictions
        self.preds_per_timepoint[timepoint].append(pos_pred.detach().cpu())
        # Update the ground truth cells only if needed
        if self.gts_per_timepoint[timepoint] is None:
            self.gts_per_timepoint[timepoint] = pos_gt.detach().cpu()

    def compute(self) -> dict[str, torch.Tensor]:
        if any(map(lambda x: x is None, self.gts_per_timepoint.values())) or any(
            map(lambda x: len(x) == 0, self.preds_per_timepoint.values())
        ):
            raise ValueError("Did you call compute before update?")

        # Compute per timepoint and then take the mean and max
        dists: list[torch.Tensor] = []
        for timepoint in self.preds_per_timepoint.keys():
            preds = torch.cat(self.preds_per_timepoint[timepoint], dim=0).to(self.device)
            gts = self.gts_per_timepoint[timepoint].to(self.device)

            _, dist = nn_of_x_in_y(x=gts, y=preds, chunk_size=self.chunk_size)
            dists.append(dist.detach().cpu())

        dists = torch.cat(dists)

        return {
            f"{self.prefix}/spd/mean": torch.mean(dists),
            f"{self.prefix}/spd/max": torch.max(dists),
        }


class FlowMatching(LightningModule):
    def __init__(
        self,
        flow: PointCloudFlow | SinglePointFlow,
        classifier: CTClassifierNet,
        classifier_ckpt_path: str,
        optimizer: type[Optimizer],
        lr_scheduler: type[LRScheduler] | None = None,
        lr_scheduler_args: dict[str, Any] | None = None,
        nn_chunk_size: int = 5000,
        spd_chunk_size: int = 5000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["flow", "classifier", "optimizer", "lr_scheduler"])

        self.flow = flow
        self.classifier = classifier
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args
        self.nn_chunk_size = nn_chunk_size

        # Load the classifier from a torch lightning checkpoint.
        load_classifier(classifier=self.classifier, ckpt_path=classifier_ckpt_path)
        self.softmax = torch.nn.Softmax(dim=-1)

        # Number of slices
        self.n_slices = self.flow.backbone.ohe_dim

        # 1-Nearest Neighbor classification metrics
        self.n_cell_types = self.classifier.output_dim
        nn_class_metrics = get_nn_class_metrics(num_classes=self.n_cell_types)

        # Point-to-Shape distance
        psd = get_psd_metrics()

        # Regression metrics
        reg_metrics = get_reg_metrics()

        # Validation metrics
        self.val_nn_class_metrics = nn_class_metrics.clone(prefix="val/nn_class/")
        self.val_psd_metrics = psd.clone(prefix="val/psd/")
        self.val_x_reg_metrics = reg_metrics.clone(prefix="val/x/")
        self.val_pos_reg_metrics = reg_metrics.clone(prefix="val/pos/")
        self.val_spd_metrics = ShapeToPointDistance(
            n_slices=self.n_slices - 1, device=self.device, prefix="val", chunk_size=spd_chunk_size
        )

        # Test metrics
        self.test_nn_class_metrics = nn_class_metrics.clone(prefix="test/nn_class/")
        self.test_psd_metrics = psd.clone(prefix="test/psd/")
        self.test_x_reg_metrics = reg_metrics.clone(prefix="test/x/")
        self.test_pos_reg_metrics = reg_metrics.clone(prefix="test/pos/")
        self.test_spd_metrics = ShapeToPointDistance(
            n_slices=self.n_slices - 1, device=self.device, prefix="test", chunk_size=spd_chunk_size
        )

    def training_step(self, batch: STTrainDataBatch, _) -> FlowLosses:
        losses = self.flow.loss(batch=batch)
        self.log_dict(
            {f"train/{key}": value.item() for key, value in losses.items()}, prog_bar=True
        )
        return losses

    def on_validation_epoch_start(self) -> None:
        # Move the classifier to the proper device
        self.classifier = self.classifier.to(self.device)
        self.val_spd_metrics.reset()

    def on_test_epoch_start(self) -> None:
        # Move the classifier to the proper device
        self.classifier = self.classifier.to(self.device)
        self.test_spd_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        spd_metrics = self.val_spd_metrics.compute()
        self.log_dict(spd_metrics, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        spd_metrics = self.test_spd_metrics.compute()
        self.log_dict(spd_metrics, on_epoch=True)

    def eval_step(
        self,
        batch: STValDataItem,
        x_reg_metrics: tm.MetricCollection,
        pos_reg_metrics: tm.MetricCollection,
        nn_class_merics: tm.MetricCollection,
        psd_metrics: tm.MetricCollection,
        spd_metrics: ShapeToPointDistance,
    ) -> None:
        x_traj, pos_traj = self.flow.sample(batch=batch)
        x_t2_pred, pos_t2_pred = x_traj[-1], pos_traj[-1]
        x_t2_gt, pos_t2_gt = batch["X_t2"], batch["pos_t2"]

        # Compute regression metrics
        x_metrics, pos_metrics = (
            x_reg_metrics(x_t2_gt.contiguous(), x_t2_pred.contiguous()),
            pos_reg_metrics(pos_t2_gt.contiguous(), pos_t2_pred.contiguous()),
        )

        # Compute classification metrics
        classes_preds = (
            torch.argmax(self.softmax(self.classifier(x_t2_pred)), dim=-1).to(torch.long).view(-1)
        )

        # Find closest real neighbor of each predicted cell and keep the computed distances
        # which we can use to compute the Point-to-Shape distance
        idxs, dists = nn_of_x_in_y(
            # (N_pred, pos_dim)
            x=pos_t2_pred.view(-1, pos_t2_pred.size(-1)),
            # (N_gt, pos_dim)
            y=batch["global_pos_t2"],
            chunk_size=self.nn_chunk_size,
        )

        # Compute the classification metrics
        class_metrics = nn_class_merics(classes_preds, batch["global_ct_t2"][idxs])

        # Compute the Point-to-Shape distance
        psd = psd_metrics(dists)

        # Computing the Shape-to-Point distance does not work on per step basis
        # as we need to first accumulate all predictions that we make for a timepoint
        # We can only then find the closest neighbor in the predicted data for each
        # ground truth cell.
        # The "t2_ohe" is always of shape (N_microenvs | 1, N_points, n_slices)
        # We substract 1 from the timepoint as it begins at 1 since it is the t2 timepoint.
        timepoint = batch["t2_ohe"][0, 0].nonzero().item() - 1
        # Accumulate
        spd_metrics.update(
            # (N_pred, pos_dim)
            pos_pred=pos_t2_pred.view(-1, pos_t2_pred.size(-1)),
            # (N_gt, pos_dim)
            pos_gt=batch["global_pos_t2"],
            timepoint=timepoint,
        )

        # Log all metrics (The spd metrics will be logged in the on epoch end hooks)
        self.log_dict({**x_metrics, **pos_metrics, **class_metrics, **psd}, on_epoch=True)

    def validation_step(self, batch: STValDataItem, _) -> None:
        self.eval_step(
            batch=batch,
            x_reg_metrics=self.val_x_reg_metrics,
            pos_reg_metrics=self.val_pos_reg_metrics,
            nn_class_merics=self.val_nn_class_metrics,
            psd_metrics=self.val_psd_metrics,
            spd_metrics=self.val_spd_metrics,
        )

    def test_step(self, batch: STValDataItem, _) -> None:
        self.eval_step(
            batch=batch,
            x_reg_metrics=self.test_x_reg_metrics,
            pos_reg_metrics=self.test_pos_reg_metrics,
            nn_class_merics=self.test_nn_class_metrics,
            psd_metrics=self.test_psd_metrics,
            spd_metrics=self.test_spd_metrics,
        )

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer(
            params=self.flow.parameters(),
        )
        config = {"optimizer": optimizer}
        if self._lr_scheduler is not None and self._lr_scheduler_args is not None:
            config["lr_scheduler"] = {
                "scheduler": self._lr_scheduler(optimizer=optimizer),
                **self._lr_scheduler_args,
            }
        return config

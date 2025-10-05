from typing import Any

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
            "auc": tm.AUROC(task="multiclass", num_classes=num_classes),
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


class ShapeToPointDistance:
    def __init__(self, n_slices: int, device: torch.device) -> None:
        self.n_slices = n_slices
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.acc_pred: dict[int, list[torch.Tensor]] = {i: [] for i in range(self.n_slices)}
        self.acc_gt: dict[int, list[torch.Tensor]] = {i: [] for i in range(self.n_slices)}

    def update(self) -> None:
        # Update the predictions
        # Update the ground truth cells only if needed
        pass

    def compute(self) -> dict[str, torch.Tensor]:
        # Well, compute per timepoint and then take the mean and max
        pass


class FlowMatching(LightningModule):
    def __init__(
        self,
        flow: PointCloudFlow | SinglePointFlow,
        classifier: CTClassifierNet,
        optimizer: type[Optimizer],
        classifier_ckpt_path: str,
        lr_scheduler: type[LRScheduler] | None = None,
        lr_scheduler_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.classifier = classifier
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args

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

        # Shape-to-Point distance
        spd = ...

        # Regression metrics
        reg_metrics = get_reg_metrics()

        # Validation metrics
        self.val_nn_class_metrics = nn_class_metrics.clone(prefix="val/nn_class/")
        self.val_psd_metrics = psd.clone(prefix="val/psd/")
        self.val_x_reg_metrics = reg_metrics.clone(prefix="val/x/")
        self.val_pos_reg_metrics = reg_metrics.clone(prefix="val/pos/")

        # Test metrics
        self.test_nn_class_metrics = nn_class_metrics.clone(prefix="test/nn_class/")
        self.test_psd_metrics = psd.clone(prefix="test/psd/")
        self.test_x_reg_metrics = reg_metrics.clone(prefix="test/x/")
        self.test_pos_reg_metrics = reg_metrics.clone(prefix="test/pos/")

    def training_step(self, batch: STTrainDataBatch, _) -> FlowLosses:
        losses = self.flow.loss(batch=batch)
        self.log_dict(
            {f"train/{key}": value.item() for key, value in losses.items()}, prog_bar=True
        )
        return losses

    def on_validation_epoch_start(self) -> None:
        # Move the classifier to the proper device
        self.classifier = self.classifier.to(self.device)

    def eval_step(
        self,
        batch: STValDataItem,
        x_reg_metrics: tm.MetricCollection,
        pos_reg_metrics: tm.MetricCollection,
        nn_class_merics: tm.MetricCollection,
        psd_metrics: tm.MetricCollection,
    ) -> None:
        x_traj, pos_traj = self.flow.sample(batch=batch)
        x_t2_pred, pos_t2_pred = x_traj[-1], pos_traj[-1]
        x_t2_gt, pos_t2_gt = batch["X_t2"], batch["pos_t2"]

        # Compute regression metrics
        x_metrics, pos_metrics = (
            x_reg_metrics(x_t2_gt, x_t2_pred),
            pos_reg_metrics(pos_t2_gt, pos_t2_pred),
        )

        # Compute classification metrics
        classes_preds = torch.argmax(self.softmax(self.classifier(x_t2_pred)), dim=-1).to(
            torch.long
        )

        # Find closest real neighbor of each predicted cell and keep the computed distances
        # which we can use to compute the Point-to-Shape distance
        # idxs, distances = closest_neighbor(pos_t2_pred, pos_t2_all)

        # Compute the classification metrics
        # nn_class(classes_preds, ct_t2_all[idxs])

        # Compute the Point-to-Shape distance
        # psd(distances)

        # Computing the Shape-to-Point distance does not work on per step basis
        # as we need to first accumulate all predictions that we make for a timepoint
        # We can only then find the closest neighbor in the predicted data for each
        # ground truth cell.
        # accumulate(pos_t2_pred, global_pos_t2)
        # on_validation_epoch_end(): compute_spd(); log_dict();

        # Log all metrics
        self.log_dict({**x_metrics, **pos_metrics})

    def validation_step(self, batch: STValDataItem, _) -> None:
        self.eval_step(
            batch=batch,
            x_reg_metrics=self.val_x_reg_metrics,
            pos_reg_metrics=self.val_pos_reg_metrics,
            nn_class_merics=self.val_nn_class_metrics,
            psd_metrics=self.val_psd_metrics,
        )

    def test_step(self, batch: STValDataItem, _) -> None:
        self.eval_step(
            batch=batch,
            x_reg_metrics=self.test_x_reg_metrics,
            pos_reg_metrics=self.test_pos_reg_metrics,
            nn_class_merics=self.test_nn_class_metrics,
            psd_metrics=self.test_psd_metrics,
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

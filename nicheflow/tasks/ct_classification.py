from collections.abc import Sequence
from typing import Any

import torchmetrics as tm
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nicheflow.models import CTClassifierNet
from nicheflow.utils import render_and_close


# Copied from https://github.com/martenlienen/unhippo/blob/main/unhippo/tasks/classification.py
@render_and_close
def plot_tm_metric(metric: tm.Metric):
    fig, _ = metric.plot()
    return fig


class Plots(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, task: LightningModule) -> None:
        wandb_logger: WandbLogger | None = trainer.logger
        if wandb_logger is None:
            return

        plots = [
            (metric_name, plot_tm_metric(metric))
            for metric_name, metric in task.val_plot_metrics.items()
        ]
        wandb_logger.log_image(
            key="val_plots",
            images=[img for _, img in plots],
            step=trainer.global_step,
            caption=[caption.removeprefix("Multiclass") for (caption, _) in plots],
        )
        return super().on_validation_epoch_end(trainer, task)

    def on_test_epoch_end(self, trainer: Trainer, task: LightningModule) -> None:
        wandb_logger: WandbLogger | None = trainer.logger
        if wandb_logger is None:
            return

        plots = [
            (metric_name, plot_tm_metric(metric))
            for metric_name, metric in task.test_plot_metrics.items()
        ]
        wandb_logger.log_image(
            key="test_plots",
            images=[img for _, img in plots],
            step=trainer.global_step,
            caption=[caption.removeprefix("Multiclass") for (caption, _) in plots],
        )
        return super().on_test_epoch_end(trainer, task)


class CellTypeClassification(LightningModule):
    def __init__(
        self,
        net: CTClassifierNet,
        optimizer: type[Optimizer],
        lr_scheduler: type[LRScheduler] | None = None,
        lr_scheduler_args: dict[str, Any] | None = None,
        plot_callbacks: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "optimizer", "lr_scheduler"])

        self.net = net
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args
        self.plot_callbacks = plot_callbacks

        self.loss = nn.CrossEntropyLoss()

        metrics = tm.MetricCollection(
            {
                "f1": tm.F1Score(
                    task="multiclass", num_classes=self.net.output_dim, average="weighted"
                ),
                "auc": tm.AUROC(task="multiclass", num_classes=self.net.output_dim),
                "accuracy": tm.Accuracy(task="multiclass", num_classes=self.net.output_dim),
                "top3_acc": tm.Accuracy(
                    task="multiclass",
                    num_classes=self.net.output_dim,
                    top_k=3,
                    average="weighted",
                ),
                "precision": tm.Precision(
                    task="multiclass", num_classes=self.net.output_dim, average="weighted"
                ),
                "recall": tm.Recall(
                    task="multiclass", num_classes=self.net.output_dim, average="weighted"
                ),
            }
        )
        plot_metrics = tm.MetricCollection(
            {
                "ConfusionMatrix": tm.ConfusionMatrix(
                    task="multiclass", num_classes=self.net.output_dim
                ),
                "ROC": tm.ROC(task="multiclass", num_classes=self.net.output_dim),
                "PRCurve": tm.PrecisionRecallCurve(
                    task="multiclass", num_classes=self.net.output_dim
                ),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.val_plot_metrics = plot_metrics.clone(prefix="val/")
        self.test_plot_metrics = plot_metrics.clone(prefix="test/")

    def training_step(self, batch: dict[str, Tensor], _) -> dict[str, Tensor]:
        logits = self.net(batch["X"])
        loss = self.loss(logits, batch["y"])

        self.log("train/loss", loss.item(), batch_size=batch["X"].size(0), prog_bar=True)

        return {"loss": loss}

    def eval_step(
        self,
        batch: dict[str, Tensor],
        metrics: tm.MetricCollection,
        plot_metrics: tm.MetricCollection,
    ) -> None:
        logits = self.net(batch["X"])
        self.log_dict(metrics(logits, batch["y"]))
        plot_metrics.update(logits, batch["y"])

    def validation_step(self, batch: dict[str, Tensor], _) -> None:
        self.eval_step(batch, self.val_metrics, self.val_plot_metrics)

    def test_step(self, batch: dict[str, Tensor], _) -> None:
        self.eval_step(batch, self.test_metrics, self.test_plot_metrics)

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer(
            params=self.net.parameters(),
        )
        config = {"optimizer": optimizer}
        if self._lr_scheduler is not None and self._lr_scheduler_args is not None:
            config["lr_scheduler"] = {
                "scheduler": self._lr_scheduler(optimizer=optimizer),
                **self._lr_scheduler_args,
            }
        return config

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        callbacks = super().configure_callbacks()
        if self.plot_callbacks:
            callbacks = [*callbacks, Plots()]
        return callbacks

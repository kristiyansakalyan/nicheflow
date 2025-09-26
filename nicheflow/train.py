import hydra
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from nicheflow.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)

_logger = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig) -> float | None:
    set_seed(config)

    # Resolve
    OmegaConf.resolve(config)
    print_config(config)

    _logger.info(f"Instantiating datamodule <{config.data.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.data.datamodule)

    _logger.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = instantiate(config.model)

    _logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(config.get("callbacks"))

    _logger.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(config.get("logger"))

    _logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        _logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if config.get("train"):
        _logger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if config.get("test"):
        _logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            _logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        _logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    _logger.info(metric_dict)

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()

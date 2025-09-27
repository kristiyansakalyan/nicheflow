import logging
from collections.abc import Mapping
from typing import Any

import rich
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class RankedLogger(logging.LoggerAdapter):
    """
    A multi-GPU-friendly Python logger that prefixes log messages with the process rank.

    This logger supports both all-rank logging and rank-zero-only logging, and can be used
    seamlessly across distributed training setups.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Defaults to ``__name__``.
    rank_zero_only : bool, optional
        If True, logs only from rank 0. Defaults to False.
    extra : Mapping[str, object], optional
        Additional context passed to the logger. Defaults to None.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: int | None = None, *args, **kwargs) -> None:
        """
        Log a message at a given level, optionally filtering by rank.

        Parameters
        ----------
        level : int
            Logging level (e.g., logging.INFO, logging.DEBUG).
        msg : str
            The message to log.
        rank : int, optional
            If specified, logs only from this rank. Defaults to None.
        *args
            Additional positional arguments to pass to the logger.
        **kwargs
            Additional keyword arguments to pass to the logger.

        Raises
        ------
        RuntimeError
            If the `rank_zero_only.rank` has not been set before calling this method.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)

            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            elif rank is None:
                self.logger.log(level, msg, *args, **kwargs)
            elif current_rank == rank:
                self.logger.log(level, msg, *args, **kwargs)


@rank_zero_only
def print_config(config: DictConfig) -> None:
    """
    Pretty-print a Hydra/OMEGACONF config to the console (rank-zero only).

    Parameters
    ----------
    config : DictConfig
        The configuration to display.
    """
    content = OmegaConf.to_yaml(config, resolve=True)
    rich.print(Syntax(content, "yaml"))


_logger = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """
    Log selected hyperparameters and model stats to Lightning loggers (rank-zero only).

    In addition to saving config sections, it also computes and logs:
      - Total number of model parameters
      - Trainable and non-trainable parameter counts

    Parameters
    ----------
    object_dict : dict of str to Any
        A dictionary expected to contain:
            - ``"cfg"`` : DictConfig
                The full Hydra config.
            - ``"model"`` : pl.LightningModule
                The Lightning model whose parameters will be counted.
            - ``"trainer"`` : pl.Trainer
                The Lightning trainer used for logging.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        _logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # Log parameter counts
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Log selected config sections
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["task_name"] = cfg.get("task_name")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Forward to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

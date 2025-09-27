import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from nicheflow.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """
    Instantiate Lightning callbacks from a Hydra configuration.

    This utility reads a DictConfig of callbacks (each with a `_target_` key) and uses
    Hydra to instantiate them. It gracefully handles empty or invalid configurations.

    Parameters
    ----------
    callbacks_cfg : DictConfig
        Hydra configuration containing callback definitions. Each item should
        be a DictConfig with a `_target_` field specifying the callback class.

    Returns
    -------
    list of Callback
        List of instantiated Lightning callbacks.

    Raises
    ------
    TypeError
        If `callbacks_cfg` is not a DictConfig.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        _logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            _logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """
    Instantiate Lightning loggers from a Hydra configuration.

    This utility reads a DictConfig of loggers (each with a `_target_` key) and uses
    Hydra to instantiate them. It handles empty configs or missing `_target_` keys gracefully.

    Parameters
    ----------
    logger_cfg : DictConfig
        Hydra configuration containing logger definitions. Each item should
        be a DictConfig with a `_target_` field specifying the logger class.

    Returns
    -------
    list of Logger
        List of instantiated Lightning loggers.

    Raises
    ------
    TypeError
        If `logger_cfg` is not a DictConfig.
    """
    loggers: list[Logger] = []

    if not logger_cfg:
        _logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            _logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

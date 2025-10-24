from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from omegaconf import OmegaConf

from nicheflow.tasks import FlowMatching
from nicheflow.utils import RankedLogger, manual_seed

_logger = RankedLogger(__name__, rank_zero_only=True)

# Initalize hydra
initialize(config_path="../configs", version_base=None)

# Register resolvers for OmegaConf
OmegaConf.register_new_resolver("add", lambda x, y: x + y)

# Define some constants
base_experiment_path = Path("configs/experiment")
base_ckpt_path = Path("ckpts/ot_ablations")
base_output_path = Path("outputs/eval_ot_ablations")
models = ["nicheflow", "rpcflow"]
variants = ["cfm", "gvfm", "glvfm"]
datasets = ["med", "abd", "mba"]
# 0.1 is already used in the main experiments
ot_lambdas = [0.25, 0.5, 0.75]
model_to_ckpt = {"nicheflow": "NicheFlow", "rpcflow": "RPCFlow"}

# Evaluation
eval_runs = 5


def evaluate(experiment_override: str, ot_lambda_override: float, ckpt_path: Path):
    config = compose(
        config_name="train",
        overrides=[
            f"experiment={experiment_override}",
            f"data.datamodule.ot_lambda={ot_lambda_override}",
        ],
    )
    # Overwrite the paths to deal with the hydra instance issue
    config.paths.output_dir = Path(config.paths.root_dir).joinpath("outputs")
    config.paths.work_dir = config.paths.root_dir

    OmegaConf.resolve(config)

    _logger.info(f"Instantiating datamodule <{config.data.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(config.data.datamodule)

    _logger.info(f"Instantiating model <{config.model._target_}>")
    model: FlowMatching = instantiate(config.model)

    # Load the state dictionary checkpoint
    ckpt = torch.load(
        ckpt_path,
        weights_only=False,
        map_location="cpu",
    )
    model.flow.backbone.load_state_dict(ckpt)

    _logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=None, logger=False, accelerator="auto"
    )

    result_list = []

    # Evaluate
    for run_id in range(eval_runs):
        _logger.info(f"Evaluation run {run_id + 1}/{eval_runs}")
        manual_seed(int(config.seed) + run_id)

        results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=None,
        )

        result_list.append({**results[0], "run_id": run_id})

    df = pd.DataFrame(result_list)
    base_output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        base_output_path.joinpath(f"{config.logger.wandb.name}_OTLambda={ot_lambda_override}.csv"),
        index=False,
    )


def main() -> None:
    for model in models:
        for variant in variants:
            for dataset in datasets:
                for ot_lambda in ot_lambdas:
                    config_path = base_experiment_path.joinpath(model, variant, f"{dataset}.yaml")
                    ckpt_path = base_ckpt_path.joinpath(
                        f"{model_to_ckpt[model]}_{variant.upper()}_{dataset.upper()}_OTLambda={ot_lambda}.ckpt"
                    )
                    if not config_path.exists():
                        raise FileNotFoundError(f"Cannot find configuration file {config_path}")

                    if not ckpt_path.exists():
                        raise FileNotFoundError(f"Cannot find checkpoint path {ckpt_path}")

                    evaluate(
                        experiment_override=f"{model}/{variant}/{dataset}",
                        ot_lambda_override=ot_lambda,
                        ckpt_path=ckpt_path,
                    )


if __name__ == "__main__":
    main()

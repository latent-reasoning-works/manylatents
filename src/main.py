import logging

import hydra
from omegaconf import DictConfig

import src  # noqa: F401
from src.experiment import instantiate_datamodule, instantiate_trainer, run_pipeline

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base="1.2",
)

def main(cfg: DictConfig):
    """
    Main entry point. High-level logic:
      - Instantiate the datamodule
      - Instantiate the trainer
      - Call run_pipeline to handle DR, training, evaluation, etc.
    """
    logger.info(f"Running pipeline in '{cfg.mode}' mode.")

    # 1) Instantiate data module
    datamodule = instantiate_datamodule(cfg)

    # 2) Instantiate the trainer (Lightning trainer + callbacks, loggers, etc.)
    trainer = instantiate_trainer(cfg)

    # 3) Hand off control to run_pipeline
    run_pipeline(cfg, datamodule, trainer)


if __name__ == "__main__":
    main()
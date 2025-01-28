import logging

import hydra
from omegaconf import DictConfig

import src  # noqa: F401
from src.experiment import (
    instantiate_algorithm,
    instantiate_datamodule,
    instantiate_trainer,
    run_pipeline,
)

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main entry point:
      - Instantiate datamodule
      - Instantiate trainer
      - Hand off to run_pipeline
    """
    logger.info("Starting the experiment pipeline...")
    datamodule = instantiate_datamodule(cfg)
    algorithm = instantiate_algorithm(cfg)
    trainer = instantiate_trainer(cfg)
    run_pipeline(cfg, datamodule, trainer)


if __name__ == "__main__":
    main()

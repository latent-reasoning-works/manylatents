import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import src  # noqa: F401
from src.experiment import (
    instantiate_algorithm,
    instantiate_datamodule,
    instantiate_trainer,
    run_pipeline,
)

from src.configs import register_configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register_configs()

@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point:
      - Instantiate datamodule, algorithm, trainer
      - Hand off to run_pipeline
    """
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))
    logger.info("Starting the experiment pipeline...")
    
    logger.info("Instantiating the datamodule...")
    datamodule = instantiate_datamodule(cfg)
    logger.info(f"Datamodule instance: {datamodule} (type: {type(datamodule)})")
    
    logger.info("Instantiating the algorithm...")
    embeddings, algorithm = instantiate_algorithm(cfg, datamodule=datamodule)
    
    logger.info("Instantiating the trainer...")
    trainer = instantiate_trainer(cfg)
    
    logger.info("Running the pipeline...")
    run_pipeline(cfg, datamodule, trainer, algorithm=algorithm, embeddings=embeddings)

if __name__ == "__main__":
    main()

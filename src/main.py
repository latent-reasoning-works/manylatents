import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import src  # noqa: F401
from src.configs import register_configs
from src.experiment import (
    evaluate_model,
    instantiate_algorithm,
    instantiate_datamodule,
    instantiate_trainer,
    train_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register_configs()

@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point:
      - Instantiates datamodule, algorithm, and trainer (if needed)
      - Runs Dimensionality Reduction (DR) if specified and obtains embeddings
      - Runs training and evaluation only if a neural network is provided
      
    Note: For network-only configurations, embeddings remain None at instantiation 
    and should be extracted later from the trained model.
    """
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))
    logger.info("Starting the experiment pipeline...")

    # Instantiate the datamodule
    logger.info("Instantiating the datamodule...")
    datamodule = instantiate_datamodule(cfg)
    logger.info(f"Datamodule instance: {datamodule} (type: {type(datamodule)})")

    embeddings, model = None, None

    # Run DR if configured
    if "dimensionality_reduction" in cfg.algorithm and cfg.algorithm.dimensionality_reduction is not None:
        logger.info("Running Dimensionality Reduction (DR)...")
        embeddings, _ = instantiate_algorithm(cfg, datamodule=datamodule)
        logger.info(f"DR completed. Embedding shape: {embeddings.shape if embeddings is not None else 'N/A'}")
    else:
        logger.info("No DR algorithm specified. Proceeding with raw/precomputed data.")

    # Run training and evaluation if a neural network is configured
    if "network" in cfg.algorithm and cfg.algorithm.network is not None:
        logger.info("Instantiating Neural Network model...")
        # Instantiate only the NN component; embeddings remain None at this stage
        _, model = instantiate_algorithm(cfg, datamodule=datamodule)

        logger.info("Instantiating the trainer...")
        trainer = instantiate_trainer(cfg)

        logger.info("Running training...")
        train_model(cfg, trainer, datamodule, model, embeddings)

        logger.info("Running evaluation...")
        evaluate_model(cfg, trainer, datamodule, model, embeddings)
    else:
        logger.info("No neural network specified. Skipping training and evaluation.")

    logger.info("Experiment complete.")

if __name__ == "__main__":
    main()

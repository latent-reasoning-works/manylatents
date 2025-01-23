import logging
from typing import Optional, Union

import hydra
import numpy as np
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def instantiate_algorithm(
    cfg: DictConfig,
    stage: str,
    datamodule: Optional[Union[LightningDataModule, DataLoader]] = None,
    embeddings: Optional[np.ndarray] = None,
):
    """
    """
    if stage == "dimensionality_reduction":
        # Example DR config
        dr_config = cfg.algorithm
        dr_type = dr_config.type.lower()

        if dr_type == "pca":
            from sklearn.decomposition import PCA
            logger.info("Performing PCA on the dataset to produce embeddings...")

            # Example logic for a datamodule with train_dataloader
            data_arrays = []
            for batch in datamodule.train_dataloader():
                # Convert each batch to a NumPy array (assuming it's a simple Tensor)
                data_arrays.append(batch.cpu().numpy())
            data_np = np.concatenate(data_arrays, axis=0)

            pca = PCA(n_components=dr_config.n_components)
            return pca.fit_transform(data_np)

        else:
            raise NotImplementedError(f"DR algorithm not implemented: {dr_type}")

    elif stage == "learning":
        # Hydra will instantiate your model class (e.g., AANet) from cfg.model
        model_config = cfg.model
        return hydra.utils.instantiate(model_config)

    else:
        raise ValueError(f"Unknown stage: {stage}")

def instantiate_datamodule(cfg: DictConfig) -> Union[LightningDataModule, DataLoader]:
    """
    Dynamically instantiate the data module (or dataloader) from the config.
    """
    if "datamodule" in cfg and cfg.datamodule is not None:
        dm = hydra.utils.instantiate(cfg.datamodule)
        dm.setup()
        return dm
    if "dataloader" in cfg and cfg.dataloader is not None:
        # If you want to instantiate a simple DataLoader instead
        return hydra.utils.instantiate(cfg.dataloader)
    raise ValueError("No valid 'datamodule' or 'dataloader' found in the config.")


def instantiate_trainer(cfg: DictConfig) -> Trainer:
    """
    Dynamically instantiate the PyTorch Lightning Trainer from the config.
    Handles callbacks and loggers if specified.
    """
    callbacks = hydra.utils.instantiate(cfg.trainer.callbacks) if "callbacks" in cfg.trainer else None
    loggers = hydra.utils.instantiate(cfg.trainer.loggers) if "loggers" in cfg.trainer else None

    return Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )




def train_model(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
):
    """
    Train stage:
      - Instantiate the model to train.
      - Call trainer.fit().
      - Optionally save checkpoint.
    """
    model = instantiate_algorithm(cfg, stage="learning", embeddings=embeddings)
    trainer.fit(model, datamodule=datamodule)

    if cfg.paths.model_ckpt:
        trainer.save_checkpoint(cfg.paths.model_ckpt)


def evaluate_model(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
):
    """
    Evaluation stage:
      - Instantiate (or load) the model.
      - Call trainer.test().
    """
    model = instantiate_algorithm(cfg, stage="learning", embeddings=embeddings)
    if cfg.paths.model_ckpt:
        # If you want to load from checkpoint, you could do:
        # model = YourLightningModuleClass.load_from_checkpoint(cfg.paths.model_ckpt)
        pass

    trainer.test(model, datamodule=datamodule)


def run_pipeline(
    cfg: DictConfig,
    datamodule: Union[LightningDataModule, DataLoader],
    trainer: Trainer,
):
    """
    Orchestrates the entire pipeline, deciding what to do based on cfg.mode.
    Possible modes:
      - 'dimensionality_reduction': Only compute DR embeddings.
      - 'train': DR if needed, then train a model, maybe save checkpoints.
      - 'evaluate': Possibly do DR or load existing embeddings, then evaluate.
    """

    mode = cfg.mode
    embeddings = None

    # 1) DR if "dimensionality_reduction" or "train"
    if mode in {"dimensionality_reduction", "train"}:
        embeddings = instantiate_algorithm(
            cfg,
            stage="dimensionality_reduction",
            datamodule=datamodule
        )
        if mode == "dimensionality_reduction":
            if cfg.paths.embeddings_file:
                np.save(cfg.paths.embeddings_file, embeddings)
            return  # Stop here if DR-only

    # 2) Training
    if mode == "train":
        train_model(cfg, trainer, datamodule, embeddings)

    # 3) Evaluation
    elif mode == "evaluate":
        # If you want to load precomputed embeddings:
        # embeddings = np.load(cfg.paths.embeddings_file)
        evaluate_model(cfg, trainer, datamodule, embeddings)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

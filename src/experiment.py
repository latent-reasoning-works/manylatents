from typing import Optional, Union

import numpy as np
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def train(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
):
    # 1) Instantiate the model to train (LightningModule)
    model = instantiate_algorithm(cfg, stage="learning", embeddings=embeddings)

    # 2) Fit the model
    trainer.fit(model, datamodule=datamodule)

    # 3) Optionally: save checkpoint
    if cfg.paths.model_ckpt:
        trainer.save_checkpoint(cfg.paths.model_ckpt)


def evaluate(
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    embeddings: Optional[np.ndarray] = None,
):
    # 1) Instantiate or load your model
    model = instantiate_algorithm(cfg, stage="learning", embeddings=embeddings)
    if cfg.paths.model_ckpt:
        # If you want to load from checkpoint, you can do:
        # model = ModelClass.load_from_checkpoint(cfg.paths.model_ckpt)
        pass

    # 2) Evaluate
    trainer.test(model, datamodule=datamodule)


def instantiate_algorithm(
    cfg: DictConfig,
    stage: str,
    datamodule: Optional[Union[LightningDataModule, DataLoader]] = None,
    embeddings: Optional[np.ndarray] = None,
):
    """
    A "smart" algorithm factory. 
      - If stage == 'dimensionality_reduction', load from cfg.algorithm (for DR).
      - If stage == 'learning', load from cfg.model (for NN or other classifier/regressor).
      - If embeddings is not None and stage != 'dimensionality_reduction', we might just skip DR?
    """
    if stage == "dimensionality_reduction":
        # Example: DR config
        dr_config = cfg.algorithm
        dr_type = dr_config.type.lower()

        # Potentially handle multiple DR methods
        if dr_type == "pca":
            from sklearn.decomposition import PCA
            logger.info("Performing PCA on the dataset to produce embeddings...")
            # Example logic: you'd iterate over the datamodule or a train_dataloader
            data_arrays = []
            for batch in datamodule.train_dataloader():
                # Convert to CPU numpy (assuming your batch is a simple tensor)
                data_arrays.append(batch.cpu().numpy())
            data_np = np.concatenate(data_arrays, axis=0)

            pca = PCA(n_components=dr_config.n_components)
            return pca.fit_transform(data_np)

        # Alternatively: if dr_type == 'umap', ...
        else:
            raise NotImplementedError(f"DR algorithm not implemented: {dr_type}")

    elif stage == "learning":
        # Example: model config
        # Hydra will instantiate your model class, e.g., AANet
        model_config = cfg.model
        return hydra.utils.instantiate(model_config)

    else:
        raise ValueError(f"Unknown stage: {stage}")


def instantiate_trainer(cfg: DictConfig) -> Trainer:
    # If the user has trainer callbacks or loggers in config, instantiate them:
    callbacks = hydra.utils.instantiate(cfg.trainer.callbacks) if "callbacks" in cfg.trainer else None
    loggers = hydra.utils.instantiate(cfg.trainer.loggers) if "loggers" in cfg.trainer else None

    # Trainer arguments can come directly from cfg.trainer
    return Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )


def run_pipeline(
    cfg: DictConfig,
    datamodule: Union[LightningDataModule, DataLoader],
    trainer: Trainer,
):
    """
    Orchestrates the entire pipeline. Decides what to do based on cfg.mode.
    Also decides how to chain DR -> Learning -> Evaluation or skip steps.

    Possible modes (example):
      - 'dimensionality_reduction': compute embeddings only.
      - 'train': do DR if needed, train a model, maybe save checkpoints.
      - 'evaluate': do DR if needed or load existing embeddings, evaluate a model.
    """

    mode = cfg.mode
    embeddings = None

    # 1) If we are doing DR (or training which requires DR), call DR logic
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

    # 2) If we are training, call training logic
    if mode == "train":
        train_model(cfg, trainer, datamodule, embeddings)

    # 3) If we are evaluating, call evaluation logic
    elif mode == "evaluate":
        # Possibly we do DR first or just load precomputed embeddings:
        #   embeddings = load_precomputed_embeddings(cfg.paths.embeddings_file)
        evaluate_model(cfg, trainer, datamodule, embeddings)

    else:
        raise ValueError(f"Unsupported mode: {mode}")
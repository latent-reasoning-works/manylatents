import functools
import logging
from typing import Any, Optional, Tuple, Union

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.utils.data import DummyDataModule, subsample_data_and_dataset
from src.utils.metrics import flatten_and_unroll_metrics
from src.utils.utils import check_or_make_dirs

logger = logging.getLogger(__name__)

def instantiate_datamodule(cfg: DictConfig) -> LightningDataModule:
    check_or_make_dirs(cfg.cache_dir)
    logger.info(f"Cache directory ensured at: {cfg.cache_dir}")
    ##todo: change to use a test data config call instead, remove dummy utils call
    if cfg.data.get("debug", False):
        logger.info("DEBUG MODE: Using a dummy datamodule with limited data.")
        dummy_data = torch.randn(100, 10)
        dummy_labels = torch.zeros(100)
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        return DummyDataModule(
            dataset=dataset, 
            batch_size=cfg.data.batch_size, 
            num_workers=cfg.data.num_workers
        )

    datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}
    dm = hydra.utils.instantiate(datamodule_cfg)
    dm.setup()
    return dm

def instantiate_trainer(cfg: DictConfig) -> Trainer:
    """
    Dynamically instantiate the PyTorch Lightning Trainer from the config.
    Handles callbacks and loggers if specified.
    """
    trainer_cb_cfg = cfg.algorithm.callbacks.trainer

    callbacks = [
        hydra.utils.instantiate(cb_conf)
        for cb_conf in trainer_cb_cfg.values()
    ]

    loggers = []
    if "logger" in cfg and cfg.logger is not None:
        for lg_conf in cfg.logger.values():
            loggers.append(hydra.utils.instantiate(lg_conf))

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop("callbacks", None)
    trainer_cfg.pop("logger", None)

    return hydra.utils.instantiate(trainer_cfg, callbacks=callbacks, logger=loggers)
    
@functools.singledispatch
def evaluate(algorithm: Any, /, **kwargs) -> Tuple[str, Optional[float], dict]:
    """Evaluates the algorithm.

    Returns the name of the 'error' metric for this run, its value, and a dict of metrics.
    """
    raise NotImplementedError(
        f"There is no registered handler for evaluating algorithm {algorithm} of type "
        f"{type(algorithm)}! (kwargs: {kwargs})"
    )

@evaluate.register(dict)
def evaluate_embeddings(
    EmbeddingOutputs: dict,
    *,
    cfg: DictConfig,
    datamodule,
    **kwargs,
) -> dict:
    if EmbeddingOutputs is None or EmbeddingOutputs.get("embeddings") is None:
        logger.warning("No embeddings available for evaluation.")
        return {}

    embeddings = EmbeddingOutputs.get("embeddings")
 
    if datamodule.mode == "split":
        ds = datamodule.test_dataset
    else:
        ds = datamodule.train_dataset ## defaults to full datset on full runs

    logger.info(f"Reference data shape: {ds.data.shape}")
    logger.info(f"Computing embedding metrics for {ds.data.shape[0]} samples.")
    
    #subsample in case dataset is too large
    subsample_fraction = cfg.metrics.get("subsample_fraction", None)
    if subsample_fraction is not None:
        ds_sub, emb_sub = subsample_data_and_dataset(ds, embeddings, subsample_fraction)
        logger.info(f"Subsampled dataset to {emb_sub.shape[0]} samples.")
    else:
        ds_sub, emb_sub = ds, embeddings

    module = kwargs.get("module", None)
    
    metric_cfgs = flatten_and_unroll_metrics(cfg.metrics)
    
    results: dict[str, float] = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        results[metric_name] = metric_fn(
            embeddings=emb_sub,
            dataset=ds_sub,
            module=module,
        )
    return results


@evaluate.register(LightningModule)
def evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    cfg: DictConfig,
    trainer: Trainer,
    datamodule: Union[LightningDataModule, DataLoader],
    **kwargs,
) -> Tuple[dict, Optional[float]]:
    """
    Evaluate the LightningModule on the test set and compute additional custom metrics.
    
    Returns:
        A tuple: (combined_metrics, error_value).
        If evaluation is skipped (no test_step or empty results), returns ({}, None).
    """
    if not hasattr(algorithm, "test_step"):
        logger.info("Model does not define a test_step() method; skipping evaluation.")
        return {}, None

    results = trainer.test(model=algorithm, datamodule=datamodule)
    if not results:
        return {}, None

    base_metrics = results[0]
    custom_metrics = {}
    model_metrics_cfg = cfg.metrics.get("model", {})

    for metric_key, metric_params in model_metrics_cfg.items():
        if metric_params.get("enabled", True):
            metric_fn = hydra.utils.instantiate(metric_params)
            # Let any errors during metric computation propagate.
            name, value = metric_fn(algorithm, test_results=base_metrics)
            custom_metrics[name] = value

    combined_metrics = {**base_metrics, **custom_metrics}
    error_value = next(iter(combined_metrics.values())) if combined_metrics else None
    return combined_metrics, error_value

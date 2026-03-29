"""CLI entry point for manylatents experiments."""
import functools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import Callback, LightningDataModule, Trainer
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    wandb.init  # verify real package, not wandb/ output dir
except (ImportError, AttributeError):
    wandb = None

import manylatents.configs  # noqa: F401
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.experiment import run_experiment
from manylatents.utils.metrics import flatten_and_unroll_metrics
from manylatents.utils.utils import check_or_make_dirs, setup_logging, should_disable_wandb

# Auto-discover extension plugins (omics, etc.) before @hydra.main fires
def _discover_extensions():
    from importlib.metadata import entry_points
    from hydra.core.plugins import Plugins
    from hydra.plugins.search_path_plugin import SearchPathPlugin
    plugins = Plugins.instance()
    existing = {type(p) for p in plugins.discover(SearchPathPlugin)}
    for ep in entry_points(group="manylatents.extensions"):
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, SearchPathPlugin) and cls not in existing:
                plugins.register(cls)
        except Exception:
            pass

_discover_extensions()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private instantiation helpers (Hydra translation layer)
# ---------------------------------------------------------------------------


def _instantiate_datamodule(
    cfg: DictConfig, input_data_holder: Optional[Dict] = None
) -> LightningDataModule:
    """Instantiate a LightningDataModule from Hydra config."""
    check_or_make_dirs(cfg.cache_dir)
    logger.info(f"Cache directory ensured at: {cfg.cache_dir}")
    datamodule_cfg = {k: v for k, v in cfg.data.items() if k != "debug"}

    # Inject input_data if provided (can't be in OmegaConf due to numpy array serialization)
    if input_data_holder is not None and "data" in input_data_holder:
        datamodule_cfg["data"] = input_data_holder["data"]

    dm = hydra.utils.instantiate(datamodule_cfg)
    return dm


def _instantiate_algorithm(
    algorithm_config: DictConfig,
    datamodule: LightningDataModule | None = None,
) -> Any:
    """Instantiate the algorithm, handling partially configured objects."""
    algo_or_algo_partial = hydra.utils.instantiate(
        algorithm_config, datamodule=datamodule
    )
    if isinstance(algo_or_algo_partial, functools.partial):
        if datamodule:
            return algo_or_algo_partial(datamodule=datamodule)
        return algo_or_algo_partial()
    return algo_or_algo_partial


def _instantiate_callbacks(
    trainer_cb_cfg: Dict[str, Any] = None,
    embedding_cb_cfg: Dict[str, Any] = None,
) -> Tuple[List[Callback], List[EmbeddingCallback]]:
    """Instantiate Lightning and Embedding callbacks from config dicts."""
    trainer_cb_cfg = trainer_cb_cfg or {}
    embedding_cb_cfg = embedding_cb_cfg or {}

    lightning_cbs: List[Callback] = []
    embedding_cbs: List[EmbeddingCallback] = []

    for name, one_cfg in trainer_cb_cfg.items():
        cb = hydra.utils.instantiate(one_cfg)
        if isinstance(cb, Callback):
            lightning_cbs.append(cb)
        else:
            logger.warning(f"Ignoring non-Lightning callback '{name}'")

    for name, one_cfg in embedding_cb_cfg.items():
        cb = hydra.utils.instantiate(one_cfg)
        if isinstance(cb, EmbeddingCallback):
            embedding_cbs.append(cb)
        else:
            logger.warning(f"Ignoring non-Embedding callback '{name}'")

    return lightning_cbs, embedding_cbs


def _instantiate_trainer(
    cfg: DictConfig,
    lightning_callbacks: Optional[List] = None,
    loggers: Optional[List] = None,
) -> Trainer:
    """Hydra-instantiate cfg.trainer, manually handling callbacks and logger."""
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    # Remove Hydra meta-fields we don't want to forward
    trainer_kwargs.pop("_target_", None)
    trainer_kwargs.pop("callbacks", None)
    trainer_kwargs.pop("logger", None)

    if lightning_callbacks:
        trainer_kwargs["callbacks"] = lightning_callbacks
    if loggers:
        trainer_kwargs["logger"] = loggers

    return Trainer(**trainer_kwargs)


def _instantiate_sampling(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """Instantiate sampler objects from cfg.sampling."""
    sampling_cfg = (
        OmegaConf.to_container(cfg.sampling, resolve=True)
        if hasattr(cfg, "sampling") and cfg.sampling is not None
        else None
    )
    if sampling_cfg is None:
        return None
    return {
        name: hydra.utils.instantiate(sampler_cfg)
        for name, sampler_cfg in sampling_cfg.items()
    }


def _init_wandb(cfg: DictConfig):
    """Initialize a wandb run from config. Returns the run or None."""
    wandb_disabled = should_disable_wandb(cfg) or wandb is None
    if wandb_disabled or cfg.logger is None:
        logger.info("WandB logging disabled")
        return None
    logger.info(f"Initializing wandb logger: {OmegaConf.to_yaml(cfg.logger)}")
    return wandb.init(
        project=cfg.logger.get("project", cfg.project),
        name=cfg.logger.get("name", cfg.name),
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logger.get("mode", "online"),
        dir=os.environ.get("WANDB_DIR", "logs"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig) -> Dict[str, Any]:
    setup_logging(debug=cfg.debug, log_level=getattr(cfg, "log_level", "warning"))
    logger.info("Final Config:\n" + OmegaConf.to_yaml(cfg))

    wandb_run = _init_wandb(cfg)

    # Instantiate all objects
    datamodule = _instantiate_datamodule(cfg)

    if hasattr(cfg.algorithms, "latent") and cfg.algorithms.latent is not None:
        algorithm = _instantiate_algorithm(cfg.algorithms.latent, datamodule)
    elif hasattr(cfg.algorithms, "lightning") and cfg.algorithms.lightning is not None:
        algorithm = _instantiate_algorithm(cfg.algorithms.lightning, datamodule)
    else:
        raise ValueError("No algorithm specified in configuration")

    # Callbacks
    trainer_cb_cfg = dict(cfg.trainer.get("callbacks") or {})
    if hasattr(cfg, "callbacks") and cfg.callbacks is not None:
        extra_trainer_cbs = cfg.callbacks.get("trainer") or {}
        trainer_cb_cfg.update(extra_trainer_cbs)
    embedding_cb_cfg = (
        cfg.callbacks.get("embedding") if hasattr(cfg, "callbacks") else None
    )
    lightning_cbs, embedding_cbs = _instantiate_callbacks(
        trainer_cb_cfg, embedding_cb_cfg
    )

    # Loggers
    wandb_disabled = should_disable_wandb(cfg) or wandb is None
    loggers = []
    if not wandb_disabled and cfg.logger is not None:
        for lg_conf in cfg.trainer.get("logger", {}).values():
            loggers.append(hydra.utils.instantiate(lg_conf))

    trainer = _instantiate_trainer(
        cfg, lightning_callbacks=lightning_cbs, loggers=loggers
    )
    metrics = (
        flatten_and_unroll_metrics(cfg.metrics) if cfg.metrics is not None else None
    )
    sampling = _instantiate_sampling(cfg)

    result = run_experiment(
        datamodule=datamodule,
        algorithm=algorithm,
        trainer=trainer,
        embedding_callbacks=embedding_cbs,
        metrics=metrics,
        metrics_cfg=cfg.metrics,
        sampling=sampling,
        seed=cfg.seed,
        eval_only=getattr(cfg, "eval_only", False),
        pretrained_ckpt=getattr(cfg, "pretrained_ckpt", None),
        cache_dir=getattr(cfg, "cache_dir", None),
        wandb_run=wandb_run,
    )

    return result


if __name__ == "__main__":
    main()

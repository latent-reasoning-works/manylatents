import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import lightning
import numpy as np
import torch
try:
    import wandb
    wandb.init  # verify real package, not wandb/ output dir
except (ImportError, AttributeError):
    wandb = None
from lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
)

from manylatents.algorithms.latent.latent_module_base import LatentModule
from manylatents.callbacks.embedding.base import EmbeddingCallback
from manylatents.utils.data import determine_data_source

logger = logging.getLogger(__name__)


from manylatents.evaluate import (  # noqa: F401  -- backward compat re-exports
    _flatten_metric_result,
    extract_k_requirements,
    prewarm_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_precomputed_from_datamodule(datamodule: LightningDataModule) -> Optional[Dict[str, Any]]:
    """Load precomputed embeddings from a datamodule's configured path.

    Looks for ``path`` or ``precomputed_path`` in ``datamodule.hparams`` and
    delegates to :func:`~manylatents.utils.utils.load_precomputed_embeddings`
    style loading (npy / csv / pt).

    Returns ``None`` when no path is configured.
    """
    hparams = getattr(datamodule, "hparams", {})
    precomputed_path = getattr(hparams, "precomputed_path", None) or getattr(hparams, "path", None)

    if not precomputed_path:
        return None

    ext = os.path.splitext(precomputed_path)[-1].lower()
    embeddings = None

    if ext == ".npy":
        embeddings = np.load(precomputed_path)
    elif ext == ".csv":
        delimiter = ","
        with open(precomputed_path, "r") as f:
            first_line = f.readline().strip()
        first_line_fields = first_line.split(delimiter)
        if any(not field.replace(".", "").replace("-", "").replace("e", "").isdigit() for field in first_line_fields):
            embeddings = np.genfromtxt(precomputed_path, delimiter=delimiter, skip_header=1)
        else:
            embeddings = np.loadtxt(precomputed_path, delimiter=delimiter)
    elif ext in [".pt", ".pth"]:
        loaded = torch.load(precomputed_path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            embeddings = loaded.numpy()
        elif isinstance(loaded, dict):
            if "embeddings" in loaded:
                emb = loaded["embeddings"]
                embeddings = emb.numpy() if hasattr(emb, "numpy") else np.array(emb)
            else:
                raise ValueError("Checkpoint dictionary does not contain 'embeddings' key.")
        else:
            raise ValueError(f"Unsupported type loaded from {precomputed_path}: {type(loaded)}")
    else:
        raise ValueError(f"Unsupported precomputed embedding file extension: {ext}")

    return {
        "embeddings": embeddings,
        "label": None,
        "metadata": None,
        "scores": None,
    }


# ---------------------------------------------------------------------------
# Hydra-free engine
# ---------------------------------------------------------------------------


def run_engine(
    datamodule: LightningDataModule,
    algorithm,
    trainer: Trainer,
    *,
    embedding_callbacks: Optional[List[EmbeddingCallback]] = None,
    metrics=None,
    metrics_cfg=None,
    sampling=None,
    seed: int = 42,
    eval_only: bool = False,
    pretrained_ckpt: Optional[str] = None,
    cache_dir: Optional[str] = None,
    wandb_run=None,
) -> Dict[str, Any]:
    """Hydra-free experiment engine.

    Runs the full experiment pipeline — seed, data extraction, optional
    pre-fit sampling, algorithm fit/transform, evaluation, callbacks, and
    wandb logging — without depending on any Hydra / OmegaConf objects.

    Args:
        datamodule: An already-instantiated LightningDataModule.
        algorithm: A :class:`LatentModule` or :class:`LightningModule` instance.
        trainer: An already-instantiated Lightning :class:`Trainer`.
        embedding_callbacks: Optional list of :class:`EmbeddingCallback` objects
            to run after embedding computation.
        metrics: ``list[str]`` of registry metric names **or**
            ``dict[str, DictConfig]`` of Hydra metric configs (from
            ``flatten_and_unroll_metrics``).  Passed to
            :func:`manylatents.evaluate.evaluate`.
        metrics_cfg: Raw ``cfg.metrics`` DictConfig for LightningModule model
            metrics (used by ``evaluate_lightningmodule``).  May be ``None``
            for LatentModule runs.
        sampling: Dict keyed by output name whose values are already-instantiated
            sampler objects.  A ``"dataset"`` key triggers pre-fit subsampling;
            other keys are forwarded to ``evaluate()`` for post-fit sampling.
        seed: Random seed (default 42).
        eval_only: If ``True``, skip fit/transform and load precomputed
            embeddings from the datamodule.
        pretrained_ckpt: Optional path to a pretrained checkpoint
            (LightningModule only).
        cache_dir: Optional directory for disk-persisted kNN caches.
        wandb_run: An already-initialized wandb run object. If provided,
            scalar metrics are logged and the run is finished on exit.

    Returns:
        LatentOutputs dict with keys ``"embeddings"``, ``"label"``,
        ``"metadata"``, ``"scores"``, and optionally ``"callback_outputs"``.
    """
    from manylatents.evaluate import evaluate as _evaluate

    # ---- 1. Seed ----
    lightning.seed_everything(seed, workers=True)

    # ---- 2. Data setup ----
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()
    field_index, data_source = determine_data_source(train_loader)

    results: Dict[str, Any] = {}

    # ---- 3. Eval-only path ----
    if eval_only:
        logger.info("Evaluation-only mode: loading precomputed embeddings from datamodule.")
        results = _load_precomputed_from_datamodule(datamodule) or {}
    else:
        # ---- 4a. Unroll dataloaders to tensors ----
        train_tensor = torch.cat([b[field_index].cpu() for b in train_loader], dim=0)
        test_tensor = torch.cat([b[field_index].cpu() for b in test_loader], dim=0)

        # ---- 4b. Extract train labels if available ----
        train_labels = None
        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "get_labels"):
            labels = train_dataset.get_labels()
            if labels is not None:
                train_labels = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                logger.info(f"Extracted {len(train_labels)} training labels for supervised learning")

        # ---- 4c. Pre-fit sampling ----
        pre_fit_indices = None
        if sampling is not None and "dataset" in sampling:
            dataset_sampler = sampling["dataset"]
            pre_fit_indices = dataset_sampler.get_indices(
                train_tensor.numpy() if torch.is_tensor(train_tensor) else train_tensor
            )
            train_tensor = train_tensor[pre_fit_indices]
            test_tensor = test_tensor[pre_fit_indices]
            if train_labels is not None:
                train_labels = train_labels[pre_fit_indices]
            logger.info(f"Pre-fit sampling: {len(pre_fit_indices)} samples using {type(dataset_sampler).__name__}")

        logger.info(
            f"Running algorithm on {data_source}:\n"
            f"Train tensor shape: {train_tensor.shape}\n"
            f"Test tensor shape: {test_tensor.shape}\n"
            f"Algorithm: {type(algorithm).__name__}"
        )

        # ---- 4d/4e. Execute algorithm ----
        t_total_start = time.perf_counter()
        t_step_start = time.perf_counter()
        latents = None
        model_metrics: Dict[str, Any] = {}

        if isinstance(algorithm, LatentModule):
            # ---- LatentModule path ----
            algorithm.fit(train_tensor, train_labels)
            try:
                latents = algorithm.transform(test_tensor)
            except NotImplementedError:
                logger.warning(
                    f"{type(algorithm).__name__} does not support transform(). "
                    "Falling back to fit_transform() on test data (transductive mode)."
                )
                latents = algorithm.fit_transform(test_tensor)
            logger.info(f"LatentModule embedding shape: {latents.shape}")

        elif isinstance(algorithm, LightningModule):
            # ---- LightningModule path ----
            if pretrained_ckpt:
                logger.info(f"Loading pretrained model from {pretrained_ckpt}")
                algorithm = LightningModule.load_from_checkpoint(pretrained_ckpt)
            else:
                logger.info("Running training...")
                trainer.fit(algorithm, datamodule=datamodule)

            # Model evaluation (uses metrics_cfg, not the full Hydra cfg)
            logger.info("Running model evaluation.")
            model_metrics, model_error = _evaluate_lightningmodule(
                algorithm,
                trainer=trainer,
                datamodule=datamodule,
                metrics_cfg=metrics_cfg,
            )
            logger.info(f"Model evaluation completed. Error: {model_error}, Metrics: {model_metrics}")

            # Extract embeddings from encoder
            if hasattr(algorithm, "encode"):
                logger.info("Extracting embeddings using network encoder...")
                latents = algorithm.encode(test_tensor)
                latents = latents.detach().cpu().numpy() if isinstance(latents, torch.Tensor) else latents
                logger.info(f"LightningModule embedding shape: {latents.shape}")
            else:
                logger.warning(f"LightningModule {type(algorithm).__name__} has no 'encode' method — skipping")

        step_time = time.perf_counter() - t_step_start

        # ---- 4f. Package results ----
        if latents is not None:
            results = {
                "embeddings": latents,
                "label": getattr(getattr(datamodule, "test_dataset", None), "get_labels", lambda: None)(),
                "metadata": {
                    "source": "single_algorithm",
                    "algorithm_type": type(algorithm).__name__,
                    "data_shape": test_tensor.shape,
                    "step_time": step_time,
                },
            }

            # Merge model metrics (LightningModule path)
            if model_metrics:
                results.setdefault("scores", {}).update(model_metrics)

            # ---- 4g. Attach extra outputs from LatentModule ----
            if isinstance(algorithm, LatentModule):
                extras = algorithm.extra_outputs()
                for key, val in extras.items():
                    results[key] = val
                    shape_info = f" shape={val.shape}" if hasattr(val, "shape") else ""
                    logger.info(f"Extra output attached: {key}{shape_info}")

            # ---- 4h. Evaluate embedding metrics ----
            if metrics is not None:
                logger.info(f"Evaluating embeddings from {type(algorithm).__name__}...")
                # Build post-fit sampling dict (exclude pre-fit "dataset" key)
                post_fit_sampling = None
                if sampling is not None:
                    post_fit_sampling = {k: v for k, v in sampling.items() if k != "dataset"}
                    if not post_fit_sampling:
                        post_fit_sampling = None

                # Determine dataset for evaluation
                mode = getattr(datamodule, "mode", None) or getattr(getattr(datamodule, "hparams", None), "mode", "full")
                if mode == "split":
                    ds = datamodule.test_dataset
                else:
                    ds = datamodule.train_dataset

                t_eval_start = time.perf_counter()
                results["scores"] = _evaluate(
                    results["embeddings"],
                    dataset=ds,
                    module=algorithm if isinstance(algorithm, LatentModule) else None,
                    metrics=metrics,
                    sampling=post_fit_sampling,
                    cache_dir=cache_dir,
                )
                eval_time = time.perf_counter() - t_eval_start
                total_time = time.perf_counter() - t_total_start

                results["metadata"]["eval_time"] = eval_time
                results["metadata"]["total_time"] = total_time

    # ---- 5. Run embedding callbacks ----
    callback_outputs: Dict[str, Any] = {}
    if results and embedding_callbacks:
        for cb in embedding_callbacks:
            cb_result = cb.on_latent_end(dataset=datamodule.test_dataset, embeddings=results)
            if isinstance(cb_result, dict):
                callback_outputs.update(cb_result)
                logger.info(f"Callback {cb.__class__.__name__} returned: {list(cb_result.keys())}")

    if callback_outputs:
        results["callback_outputs"] = callback_outputs
        logger.info(f"Added callback outputs to results: {list(callback_outputs.keys())}")

    # ---- 6. Log to wandb ----
    if wandb_run is not None and results.get("scores"):
        scores = results["scores"]
        scalar_metrics = {}
        for name, val in scores.items():
            if isinstance(val, tuple) and len(val) == 2:
                scalar_metrics[f"metrics/{name}"] = float(val[0])
            elif np.ndim(val) == 0:
                scalar_metrics[f"metrics/{name}"] = float(val)
        if scalar_metrics and wandb is not None:
            wandb.log(scalar_metrics)
            logger.info(f"Auto-logged {len(scalar_metrics)} metrics to wandb: {list(scalar_metrics.keys())}")

    if wandb_run is not None and wandb is not None:
        wandb.finish()

    logger.info("Engine run complete.")

    # ---- 7. Return results ----
    return results


def _evaluate_lightningmodule(
    algorithm: LightningModule,
    *,
    trainer: Trainer,
    datamodule,
    metrics_cfg=None,
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Evaluate a LightningModule without requiring a full Hydra cfg.

    This is the Hydra-free counterpart of the old ``evaluate_lightningmodule``.
    It runs ``trainer.test()`` and then any model-level metrics specified via
    *metrics_cfg*.

    Args:
        algorithm: The LightningModule to evaluate.
        trainer: Lightning Trainer.
        datamodule: DataModule or DataLoader for testing.
        metrics_cfg: Optional metric configs (DictConfig or dict).  If provided
            and contains a ``"model"`` key, those model metrics are instantiated
            via ``hydra.utils.instantiate`` and applied.

    Returns:
        (combined_metrics, error_value) tuple.
    """
    if not hasattr(algorithm, "test_step"):
        logger.info("Model does not define a test_step() method; skipping evaluation.")
        return {}, None

    results = trainer.test(model=algorithm, datamodule=datamodule)
    if not results:
        return {}, None

    base_metrics = results[0]
    custom_metrics: Dict[str, Any] = {}

    # Model-level metrics from config (if any)
    model_metrics_cfg: Dict[str, Any] = {}
    if metrics_cfg is not None:
        if hasattr(metrics_cfg, "get"):
            model_metrics_cfg = metrics_cfg.get("model", {}) or {}
        elif isinstance(metrics_cfg, dict):
            model_metrics_cfg = metrics_cfg.get("model", {})

    for metric_key, metric_params in model_metrics_cfg.items():
        if isinstance(metric_params, dict) and not metric_params.get("enabled", True):
            continue
        if hasattr(metric_params, "get") and not metric_params.get("enabled", True):
            continue
        import hydra as _hydra
        metric_fn = _hydra.utils.instantiate(metric_params)
        name, value = metric_fn(algorithm, test_results=base_metrics)
        custom_metrics[name] = value

    combined_metrics = {**base_metrics, **custom_metrics}
    error_value = next(iter(combined_metrics.values())) if combined_metrics else None
    return combined_metrics, error_value
